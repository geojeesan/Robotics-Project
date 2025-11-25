# youbot_yolo_picker.py
# Combines YOLOv8 bottle detection with the existing state machine and arm control
# logic for a YouBot in Webots.

from controller import Robot
import cv2
import numpy as np
import math

# YOLO imports and torch safe globals
from ultralytics import YOLO
import torch
try:
    # Required for safe loading of PyTorch models within the Webots environment
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn.modules.container import Sequential
    from torch.nn import Module
    torch.serialization.add_safe_globals([DetectionModel, Sequential, Module])
except Exception:
    pass

# ===============================================================
# PARAMETERS
# ===============================================================
TIME_STEP = 64
FRAME_SKIP = 1
DEBUG_VIEW = True

# Detection
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.15
TARGET_CLASSES = [39]          # 39 = bottle (COCO dataset)
CAMERA_RESIZE = (320, 240)

# Motion & State
CAMERA_WARMUP_STEPS = 5
LOST_DETECTION_TOLERANCE = 50
STOPPING_SPEED = 0.05
MAX_FORWARD_SPEED = 0.20
KP_TURN = 0.0035               # Proportional gain for turning
TARGET_LOCK_THRESHOLD = 5      # Frames to confirm target before approach
IDEAL_STOP_Y_RATIO = 0.80      # Target vertical position for slow approach
FINAL_STOP_Y_RATIO = 0.90      # Vertical position to stop and trigger pickup

# Arm Timings (Increased to ensure enough physical time for arm movement)
# The motor velocity (0.5) and these frames determine the pick duration.
PICK_PREP_FRAMES = 80          # Time to move arm to pick position (was 60)
GRIP_CLOSE_FRAMES = 150        # Time index to finish closing gripper (was 100)
ARM_LIFT_FRAMES = 250          # Time index to finish lifting arm (was 180)
COMPLETE_RESET_FRAMES = 30     # Time to reset arm after completion

# Robot geometry for omnidirectional base
WHEEL_RADIUS = 0.05
LX = 0.228
LY = 0.158

# Device Names
WHEEL_NAMES = ["wheel1", "wheel2", "wheel3", "wheel4"]
ARM_NAMES = ["arm1", "arm2", "arm3", "arm4", "arm5"]
GRIP_LEFT = "finger::left"
GRIP_RIGHT = "finger::right" 

# ===============================================================
# INITIALIZATION
# ===============================================================
robot = Robot()

# Camera
camera = robot.getDevice("camera")
if camera is None:
    raise RuntimeError("Camera device 'camera' not found.")
camera.enable(TIME_STEP)
cam_w = camera.getWidth()
cam_h = camera.getHeight()

# Wheels
wheels = []
for name in WHEEL_NAMES:
    m = robot.getDevice(name)
    m.setPosition(float("inf"))
    m.setVelocity(0.0)
    wheels.append(m)

# Arm joints
arm_joints = []
for name in ARM_NAMES:
    try:
        j = robot.getDevice(name)
        # Setting a moderate velocity helps the arm reach the position smoothly
        j.setVelocity(0.5) 
        arm_joints.append(j)
    except:
        arm_joints.append(None)

# Gripper (Using both left and right for completeness)
has_gripper = False
gripper_left = None
gripper_right = None
try:
    gripper_left = robot.getDevice(GRIP_LEFT)
    gripper_right = robot.getDevice(GRIP_RIGHT)
    gripper_left.setVelocity(0.03)
    gripper_right.setVelocity(0.03)
    has_gripper = True
except:
    pass

# Lidar (Optional but good practice)
LIDAR_NAME = "Hokuyo URG-04LX-UG01"
lidar = robot.getDevice(LIDAR_NAME)
has_lidar = False
if lidar:
    lidar.enable(TIME_STEP)
    lidar.enablePointCloud()
    has_lidar = True
else:
    print(f"[WARN] Lidar '{LIDAR_NAME}' NOT found.")

# YOLO Model Loading
print("[INFO] Loading YOLO...")
yolo = YOLO(YOLO_MODEL)
print("[INFO] YOLO Ready.")

# ===============================================================
# HELPER FUNCTIONS
# ===============================================================
def base_move(vx, vy, omega):
    """Move the omnidirectional base with velocities vx (forward), vy (lateral), omega (rotation)"""
    speeds = [
        1 / WHEEL_RADIUS * (vx + vy + (LX + LY) * omega),
        1 / WHEEL_RADIUS * (vx - vy - (LX + LY) * omega),
        1 / WHEEL_RADIUS * (vx - vy + (LX + LY) * omega),
        1 / WHEEL_RADIUS * (vx + vy - (LX + LY) * omega)
    ]
    for i, wheel in enumerate(wheels):
        wheel.setVelocity(speeds[i])

def base_stop():
    """Stop the base"""
    base_move(0, 0, 0)

def base_turn(angular_velocity):
    """Turn in place"""
    base_move(0, 0, angular_velocity)

def base_forward(linear_velocity):
    """Move forward"""
    base_move(linear_velocity, 0, 0)

def arm_reset():
    """Reset arm to default position (folded)"""
    poses = [0.0, 1.57, -2.635, 1.78, 0.0]
    for j, p in zip(arm_joints, poses): 
        if j: j.setPosition(p)

def arm_pick_position():
    """Set arm to picking position (down and forward)"""
    poses = [0.0, -0.97, -1.55, -0.61, 0.0]
    for j, p in zip(arm_joints, poses): 
        if j: j.setPosition(p)

def arm_lift_position():
    """Set arm to lift position (raised, carrying)"""
    poses = [2.949, 0.92, 0.42, 1.78, 0.0]
    for j, p in zip(arm_joints, poses): 
        if j: j.setPosition(p)

def gripper_open():
    """Open the gripper"""
    if has_gripper:
        gripper_left.setPosition(0.025)
        gripper_right.setPosition(0.025)

def gripper_close():
    """Close the gripper to grip object"""
    if has_gripper:
        gripper_left.setPosition(0.0)
        gripper_right.setPosition(0.0)

# ===============================================================
# YOLO DETECTION FUNCTION
# ===============================================================
def detect_bottle(image):
    """Detect the largest bottle (class 39) using YOLOv8."""
    frame = np.frombuffer(image, np.uint8).reshape((cam_h, cam_w, 4))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Run YOLO on the resized frame
    results = yolo(cv2.resize(frame_bgr, CAMERA_RESIZE), 
                   conf=YOLO_CONF, 
                   classes=TARGET_CLASSES, 
                   verbose=False)
    
    boxes = results[0].boxes
    det = None
    best_area = 0
    
    if boxes:
        # Scale factor from resized frame back to camera frame
        sx, sy = cam_w / CAMERA_RESIZE[0], cam_h / CAMERA_RESIZE[1]
        
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            w = (xyxy[2]-xyxy[0]) * sx
            h = (xyxy[3]-xyxy[1]) * sy
            area = w * h
            
            # Find the largest bottle (most likely the closest one)
            if area > best_area:
                best_area = area
                cx = (xyxy[0] + xyxy[2])/2 * sx
                cy = (xyxy[1] + xyxy[3])/2 * sy
                
                # Store detected center position and area
                det = (int(cx), int(cy), int(area))
                
                # Draw box for debug view
                if DEBUG_VIEW:
                    x1, y1 = int(xyxy[0]*sx), int(xyxy[1]*sy)
                    x2, y2 = int(xyxy[2]*sx), int(xyxy[3]*sy)
                    cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame_bgr, f"Bottle Area: {int(area)}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if DEBUG_VIEW:
        # Draw target lines
        img_center_x = cam_w // 2
        ideal_y_stop = int(cam_h * IDEAL_STOP_Y_RATIO)
        final_y_stop = int(cam_h * FINAL_STOP_Y_RATIO)
        cv2.line(frame_bgr, (img_center_x, 0), (img_center_x, cam_h), (255, 0, 0), 1)
        cv2.line(frame_bgr, (0, ideal_y_stop), (cam_w, ideal_y_stop), (0, 255, 255), 1) # Yellow
        cv2.line(frame_bgr, (0, final_y_stop), (cam_w, final_y_stop), (0, 0, 255), 2)   # Red

        cv2.imshow("YouBot YOLO Vision", frame_bgr)
        cv2.waitKey(1)
        
    return det

# ===============================================================
# MAIN LOOP
# ===============================================================

arm_reset()
gripper_open()

frame_count = 0
lost_counter = 0
state = "SEARCHING"
pickup_timer = 0
target_lock_counter = 0
target_position = None
last_known_error = 0
last_known_area = 0

while robot.step(TIME_STEP) != -1:
    frame_count += 1
    if frame_count <= CAMERA_WARMUP_STEPS:
        continue
    if frame_count % FRAME_SKIP != 0:
        continue

    image = camera.getImage()
    if not image:
        continue

    # Use YOLO to detect the bottle
    detection = detect_bottle(image)

    # --- STATE MACHINE INPUTS ---
    if detection:
        center_x, center_y, area = detection
        error = center_x - cam_w // 2
        lost_counter = 0
        
        # Tracking and Locking Logic
        if target_lock_counter < TARGET_LOCK_THRESHOLD:
            target_lock_counter += 1
            if target_lock_counter == TARGET_LOCK_THRESHOLD:
                print(f"[TRACKING] Target locked.")
        
        last_known_error = error
        last_known_area = area
        
    else:
        # No detection
        lost_counter += 1
        target_lock_counter = 0

    # --- STATE MACHINE EXECUTION ---
    if state == "SEARCHING":
        if target_lock_counter >= TARGET_LOCK_THRESHOLD:
            state = "APPROACH"
            print("[STATE] -> APPROACH (Target found and locked)")
            
        else:
            base_turn(0.3) # Continuous search rotation
            
    elif state == "APPROACH":
        if detection:
            # 1. CRITICAL STOPPING CHECK (Your logic based on Y-position)
            final_y_stop = cam_h * FINAL_STOP_Y_RATIO
            if center_y >= final_y_stop:
                base_stop()
                print(f"[STATE] 🛑 STOP: Object center_y ({center_y:.0f}) passed final y stop ({final_y_stop:.0f}). Starting pickup sequence.")
                state = "PICKING"
                pickup_timer = 0
                continue 

            # 2. ALIGNMENT (Proportional control for centering)
            turn_speed = np.clip(-error * KP_TURN, -0.5, 0.5)

            # 3. FORWARD SPEED (Proportional control for distance)
            ideal_y_stop = cam_h * IDEAL_STOP_Y_RATIO
            if center_y < ideal_y_stop:
                # Accelerate towards the ideal stopping line
                speed_ratio = (ideal_y_stop - center_y) / ideal_y_stop
                forward_speed = np.clip(speed_ratio * MAX_FORWARD_SPEED, STOPPING_SPEED * 0.5, MAX_FORWARD_SPEED)
            else:
                # Slow crawl or stop if past ideal line but not yet at final stop
                forward_speed = STOPPING_SPEED * 0.3
            
            # Execute motion
            base_move(forward_speed, 0.0, turn_speed)

        else: # Lost target while in APPROACH state
            if lost_counter > LOST_DETECTION_TOLERANCE:
                base_stop()
                state = "SEARCHING"
                print("[STATE] -> SEARCHING (Target lost for too long)")
            else:
                # Perform a corrective turn based on last known error
                turn_speed = np.clip(-last_known_error * KP_TURN, -0.4, 0.4)
                base_turn(turn_speed)

    elif state == "PICKING":
        # Execute the timed arm sequence 
        
        pickup_timer += 1
        
        if pickup_timer < PICK_PREP_FRAMES:
            # Phase 1: Move arm to pick position and open gripper
            arm_pick_position()
            gripper_open()
        elif pickup_timer < GRIP_CLOSE_FRAMES:
            # Phase 2: Close the gripper (wait until arm is down)
            gripper_close()
        elif pickup_timer < ARM_LIFT_FRAMES:
            # Phase 3: Lift the arm with the object
            arm_lift_position()
        else:
            # Sequence complete
            state = "COMPLETE"
            base_stop()
            pickup_timer = 0
            print("[STATE] -> COMPLETE (Pick sequence finished)")

    elif state == "COMPLETE":
        # Reset and prepare for the next object
        pickup_timer += 1
        if pickup_timer == 1:
            arm_reset()
            gripper_open()
            target_lock_counter = 0
        elif pickup_timer > COMPLETE_RESET_FRAMES:
            state = "SEARCHING"
            pickup_timer = 0
            print("[STATE] -> SEARCHING (Reset complete, looking for new target)")
        else:
            base_stop()
    
# ===============================================================
# CLEANUP
# ===============================================================
cv2.destroyAllWindows()
print("[INFO] Controller finished.")