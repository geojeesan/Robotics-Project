from controller import Robot, Keyboard
import cv2
import numpy as np
import math

# Library Imports
from youbot_library import YoubotBase, YoubotArm, YoubotGripper, ParticleFilter, ParticleVisualizer

# YOLO imports
from ultralytics import YOLO
import torch
try:
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn.modules.container import Sequential
    from torch.nn import Module
    torch.serialization.add_safe_globals([DetectionModel, Sequential, Module])
except Exception:
    pass

# PARAMETERS

TIME_STEP = 64
FRAME_SKIP = 1
DEBUG_VIEW = True

# Detection
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.15
TARGET_CLASSES = [39]          # 39 = bottle
CAMERA_RESIZE = (320, 240)

# Motion & State
CAMERA_WARMUP_STEPS = 5
LOST_DETECTION_TOLERANCE = 50
STOPPING_SPEED = 0.05
MAX_FORWARD_SPEED = 0.20
KP_TURN = 0.0035
TARGET_LOCK_THRESHOLD = 5
IDEAL_STOP_Y_RATIO = 0.80
FINAL_STOP_Y_RATIO = 0.90
NUM_PARTICLES = 200

# Arm Timings
PICK_PREP_FRAMES = 80
GRIP_CLOSE_FRAMES = 150
ARM_LIFT_FRAMES = 250
COMPLETE_RESET_FRAMES = 30

# Manual Speed
MANUAL_SPEED = 0.3

# INITIALIZATION

robot = Robot()

# 1. Initilizing Components
base = YoubotBase(robot)
arm = YoubotArm(robot)
gripper = YoubotGripper(robot)

# 2. Initializing Sensors
camera = robot.getDevice("camera")
if camera:
    camera.enable(TIME_STEP)
    cam_w, cam_h = camera.getWidth(), camera.getHeight()
else:
    print("[ERROR] Camera not found")
    cam_w, cam_h = 320, 240 # Fallback

lidar = robot.getDevice("lidar")
if lidar: 
    lidar.enable(TIME_STEP)
    lidar.enablePointCloud()

gps = robot.getDevice("gps")
if gps: gps.enable(TIME_STEP)

keyboard = robot.getKeyboard()
keyboard.enable(TIME_STEP)

# 3. Initializing Particle Filter
pf = ParticleFilter("final_map.npy", NUM_PARTICLES)
vis = ParticleVisualizer(robot, pf.map_grid)

# 4. Initializing YOLO
print("[INFO] Loading YOLO...")
try:
    yolo = YOLO(YOLO_MODEL)
    print("[INFO] YOLO Ready.")
except Exception as e:
    print(f"[WARN] YOLO Failed to load: {e}")
    yolo = None

# YOLO DETECTION FUNCTION

def detect_bottle(image):
    global DEBUG_VIEW
    if yolo is None or image is None: return None
    
    frame = np.frombuffer(image, np.uint8).reshape((cam_h, cam_w, 4))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Run YOLO on resized frame
    results = yolo(cv2.resize(frame_bgr, CAMERA_RESIZE), 
                   conf=YOLO_CONF, classes=TARGET_CLASSES, verbose=False)
    
    det = None
    best_area = 0
    boxes = results[0].boxes
    
    if boxes:
        sx, sy = cam_w / CAMERA_RESIZE[0], cam_h / CAMERA_RESIZE[1]
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            w = (xyxy[2]-xyxy[0]) * sx
            h = (xyxy[3]-xyxy[1]) * sy
            area = w * h
            
            if area > best_area:
                best_area = area
                cx = (xyxy[0] + xyxy[2])/2 * sx
                cy = (xyxy[1] + xyxy[3])/2 * sy
                det = (int(cx), int(cy), int(area))
                
                if DEBUG_VIEW:
                    x1, y1 = int(xyxy[0]*sx), int(xyxy[1]*sy)
                    x2, y2 = int(xyxy[2]*sx), int(xyxy[3]*sy)
                    cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame_bgr, f"{int(area)}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if DEBUG_VIEW:
        ideal_y = int(cam_h * IDEAL_STOP_Y_RATIO)
        final_y = int(cam_h * FINAL_STOP_Y_RATIO)
        cv2.line(frame_bgr, (0, ideal_y), (cam_w, ideal_y), (0, 255, 255), 1)
        cv2.line(frame_bgr, (0, final_y), (cam_w, final_y), (0, 0, 255), 2)
        try:
            cv2.imshow("YOLO Vision", frame_bgr)
            cv2.waitKey(1)
        except cv2.error:
            print("[WARN] OpenCV GUI not supported (cv2.imshow failed). Disabling DEBUG_VIEW.")
            DEBUG_VIEW = False
        
    return det

# MAIN LOOP

arm.reset()
gripper.open()

frame_count = 0
lost_counter = 0
state = "SEARCHING"
pickup_timer = 0
target_lock_counter = 0
target_position = None
last_known_error = 0
last_known_area = 0
pc = 0

print("Controller Started. Mode: YOLO Autonomous + PF + Manual Override")

while robot.step(TIME_STEP) != -1:
    frame_count += 1
    
    # Manual overide inputs
    c = keyboard.getKey()
    manual_active = False
    
    if c >= 0:
        is_shift = (c & Keyboard.SHIFT)
        key = (c & ~Keyboard.SHIFT)
        
        if is_shift and c != pc:
            if key == Keyboard.UP: arm.increase_height()
            elif key == Keyboard.DOWN: arm.decrease_height()
        else:
            if key == Keyboard.UP: base.vx = MANUAL_SPEED; manual_active = True
            elif key == Keyboard.DOWN: base.vx = -MANUAL_SPEED; manual_active = True
            elif key == Keyboard.LEFT: base.vy = MANUAL_SPEED; manual_active = True
            elif key == Keyboard.RIGHT: base.vy = -MANUAL_SPEED; manual_active = True
            elif key == ord('N'): base.omega = MANUAL_SPEED; manual_active = True
            elif key == ord('M'): base.omega = -MANUAL_SPEED; manual_active = True
            elif key == ord(' '):
                base.reset(); arm.reset()
                state = "SEARCHING"; target_lock_counter = 0
    pc = c

    # Autonomous Logic (YOLO)
    if not manual_active:
        if frame_count > CAMERA_WARMUP_STEPS and frame_count % FRAME_SKIP == 0 and camera:
            image = camera.getImage()
            detection = detect_bottle(image)
            
            # Update tracking info
            if detection:
                center_x, center_y, area = detection
                error = center_x - cam_w // 2
                lost_counter = 0
                last_known_error = error
                last_known_area = area
                
                if target_lock_counter < TARGET_LOCK_THRESHOLD:
                    target_lock_counter += 1
            else:
                lost_counter += 1
                target_lock_counter = 0

            # State Machine
            if state == "SEARCHING":
                if target_lock_counter >= TARGET_LOCK_THRESHOLD:
                    state = "APPROACH"
                    print("[STATE] APPROACH")
                else:
                    base.omega = 0.3
                    base.vx = 0; base.vy = 0
            
            elif state == "APPROACH":
                if detection:
                    final_y = cam_h * FINAL_STOP_Y_RATIO
                    ideal_y = cam_h * IDEAL_STOP_Y_RATIO
                    
                    if center_y >= final_y:
                        base.stop()
                        state = "PICKING"
                        pickup_timer = 0
                        print("[STATE] PICKING")
                    else:
                        base.omega = np.clip(-error * KP_TURN, -0.5, 0.5)
                        
                        if center_y < ideal_y:
                            ratio = (ideal_y - center_y) / ideal_y
                            base.vx = np.clip(ratio * MAX_FORWARD_SPEED, STOPPING_SPEED, MAX_FORWARD_SPEED)
                        else:
                            base.vx = STOPPING_SPEED
                        base.vy = 0
                else:
                    if lost_counter > LOST_DETECTION_TOLERANCE:
                        state = "SEARCHING"
                    else:
                        base.omega = np.clip(-last_known_error * KP_TURN, -0.4, 0.4)

            elif state == "PICKING":
                pickup_timer += 1
                if pickup_timer < PICK_PREP_FRAMES:
                    arm.set_raw_pos([0.0, -0.97, -1.55, -0.61, 0.0]) # Pick
                    gripper.open()
                elif pickup_timer < GRIP_CLOSE_FRAMES:
                    gripper.grip()
                elif pickup_timer < ARM_LIFT_FRAMES:
                    arm.set_raw_pos([2.949, 0.92, 0.42, 1.78, 0.0]) # Lift
                else:
                    state = "COMPLETE"
                    base.stop()
                    pickup_timer = 0
            
            elif state == "COMPLETE":
                pickup_timer += 1
                if pickup_timer == 1:
                    arm.reset(); gripper.open()
                    target_lock_counter = 0
                elif pickup_timer > COMPLETE_RESET_FRAMES:
                    state = "SEARCHING"
                    pickup_timer = 0
                else:
                    base.stop()
        else:
            if state != "PICKING" and state != "COMPLETE":
                base.stop()
    
    # Particles
    base.update()
    
    dt = TIME_STEP / 1000.0
    if abs(base.vx) > 0.01 or abs(base.vy) > 0.01 or abs(base.omega) > 0.01:
        pf.motion_update(base.vx, base.vy, base.omega, dt)

    if gps:
        v = gps.getValues()
        pf.sensor_update_gps(v[0], v[1])
        
    if lidar:
        r = lidar.getRangeImage()
        if r: pf.sensor_update_lidar(r, lidar.getFov())

    pf.resample()
    ex, ey, eth = pf.get_estimate()
    
    gx, gy = (gps.getValues()[0], gps.getValues()[1]) if gps else (None, None)
    vis.update(pf.particles, ex, ey, eth, gx, gy)

cv2.destroyAllWindows()