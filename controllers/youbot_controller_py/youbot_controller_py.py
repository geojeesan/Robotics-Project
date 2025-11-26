from controller import Supervisor, Keyboard, Display
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
STOPPING_SPEED = 0.15          
MAX_FORWARD_SPEED = 0.60       
KP_TURN = 0.015                

TARGET_LOCK_THRESHOLD = 5
IDEAL_STOP_Y_RATIO = 0.80
FINAL_STOP_Y_RATIO = 0.90
NUM_PARTICLES = 200

# Arm Timings
PICK_PREP_FRAMES = 10 
GRIP_CLOSE_FRAMES = 20
ARM_LIFT_FRAMES = 30
COMPLETE_RESET_FRAMES = 30

# Manual Speed
MANUAL_SPEED = 0.6             

# Grid Coordinates for Tray
TRAY_SLOTS = [
    (-0.2, -0.1), # Back Row, Right
    (-0.2,  0.0), # Back Row, Center
    (-0.2,  0.1), # Back Row, Left
    (-0.1,  0.1), # Front Row, Left
    (-0.1,  0.0), # Front Row, Center
    (-0.1, -0.1)  # Front Row, Right
]

# INITIALIZATION

robot = Supervisor()

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

# 3. Initializing Displays
debug_display = robot.getDevice("display")
if debug_display:
    print("[INFO] Webots Display found. Will use for debug if OpenCV window fails.")
    w, h = debug_display.getWidth(), debug_display.getHeight()
    black_img = np.zeros((h, w, 3), dtype=np.uint8).tobytes()
    ir = debug_display.imageNew(black_img, Display.RGB, w, h)
    debug_display.imagePaste(ir, 0, 0, False)

# 4. Initializing Particle Filter
pf = ParticleFilter("final_map.npy", NUM_PARTICLES)
vis = ParticleVisualizer(robot, pf.map_grid)

# 5. Initializing YOLO
print("[INFO] Loading YOLO...")
try:
    yolo = YOLO(YOLO_MODEL)
    print("[INFO] YOLO Ready.")
except Exception as e:
    print(f"[WARN] YOLO Failed to load: {e}")
    yolo = None

# TRASH TELEPORT FUNCTIONS

def find_closest_bottle(robot_node, ignored_ids=None, max_dist=100.0):
    """Finds the closest node that matches the WaterBottle"""
    if ignored_ids is None:
        ignored_ids = set()
        
    root = robot.getRoot()
    children = root.getField("children")
    num_children = children.getCount()
    
    best_node = None
    min_dist_sq = float('inf')
    max_dist_sq = max_dist ** 2
    
    rx, ry, _ = robot_node.getPosition() 

    for i in range(num_children):
        node = children.getMFNode(i)
        
        if node is None:
            continue
            
        # Checking if node is in "already collected" list
        if node.getId() in ignored_ids:
            continue

        # Gathering Identifiers
        n_type = node.getTypeName() 
        n_def  = node.getDef()      
        n_name = ""
        
        name_field = node.getField("name")
        if name_field:
            try:
                if name_field.getType() == Supervisor.SF_STRING:
                    n_name = name_field.getSFString()
            except:
                pass

        is_bottle = False
        
        if "WaterBottle" in n_type or "Bottle" in n_type:
            is_bottle = True
        if not is_bottle and "water bottle" in n_name.lower():
            is_bottle = True
        if not is_bottle and "BOTTLE" in n_def.upper():
            is_bottle = True
        
        if is_bottle:
            bx, by, bz = node.getPosition()
            dist_sq = (bx - rx)**2 + (by - ry)**2
            
            # Distance check
            if dist_sq < max_dist_sq and dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_node = node

    if best_node:
        print(f"[INFO] Selected closest bottle (ID: {best_node.getId()}). Dist: {math.sqrt(min_dist_sq):.2f}m")
    else:
        if max_dist < 10.0:
            print(f"[WARN] No bottle found within {max_dist}m.")
        
    return best_node

def teleport_bottle_to_back(target_node, robot_node, slot_index=0):
    """Teleports the given node to the back of the robot using grid pattern."""
    if not target_node or not robot_node: 
        return

    # 1. Get Robot Position and Rotation Matrix
    rx, ry, rz = robot_node.getPosition()
    rot = robot_node.getOrientation() # 3x3 matrix as list of 9
    
    # 2. Determining Grid Position based on Slot Index
    safe_index = slot_index % len(TRAY_SLOTS)
    grid_x, grid_y = TRAY_SLOTS[safe_index]
    
    # Defining Offset (Relative to Robot)
    off_x = grid_x
    off_y = grid_y
    off_z = 0.10 # Height above tray
    
    # If we loop around, stack them higher?
    if slot_index >= len(TRAY_SLOTS):
        off_z += 0.25 
    
    # 3. Applying Rotation to Offset
    wx = rot[0]*off_x + rot[1]*off_y + rot[2]*off_z
    wy = rot[3]*off_x + rot[4]*off_y + rot[5]*off_z
    wz = rot[6]*off_x + rot[7]*off_y + rot[8]*off_z
    
    # 4. Set New Position
    new_pos = [rx + wx, ry + wy, rz + wz]
    
    # Resetting physics
    target_node.resetPhysics()
    target_node.getField("translation").setSFVec3f(new_pos)
    target_node.getField("rotation").setSFRotation([0, 0, 1, 0]) # Resetting rotation upright

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
            x1, y1 = int(xyxy[0]*sx), int(xyxy[1]*sy)
            x2, y2 = int(xyxy[2]*sx), int(xyxy[3]*sy)
            
            # Red colour Filtering to prevent Fire Extinguisher
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 >= cam_w: x2 = cam_w - 1
            if y2 >= cam_h: y2 = cam_h - 1
            
            if x2 > x1 and y2 > y1:
                roi = frame_bgr[y1:y2, x1:x2]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Lower Red
                mask1 = cv2.inRange(hsv_roi, np.array([0, 100, 50]), np.array([10, 255, 255]))
                # Upper Red
                mask2 = cv2.inRange(hsv_roi, np.array([160, 100, 50]), np.array([180, 255, 255]))
                mask = cv2.bitwise_or(mask1, mask2)
                
                red_pixels = cv2.countNonZero(mask)
                total_pixels = (x2-x1) * (y2-y1)
                red_ratio = red_pixels / total_pixels
                
                # If more than 25% of the box is strong red, assume it's a Fire Extinguisher
                if red_ratio > 0.25:
                    if DEBUG_VIEW:
                        cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,0,255), 2)
                        cv2.putText(frame_bgr, "IGNORED (RED)", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                    continue

            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area > best_area:
                best_area = area
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                det = (int(cx), int(cy), int(area))
                
                # Draw Green Box for valid bottle
                cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame_bgr, f"{int(area)}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    ideal_y = int(cam_h * IDEAL_STOP_Y_RATIO)
    final_y = int(cam_h * FINAL_STOP_Y_RATIO)
    cv2.line(frame_bgr, (0, ideal_y), (cam_w, ideal_y), (0, 255, 255), 1)
    cv2.line(frame_bgr, (0, final_y), (cam_w, final_y), (0, 0, 255), 2)

    if DEBUG_VIEW:
        try:
            cv2.imshow("YOLO Vision", frame_bgr)
            cv2.waitKey(1)
        except cv2.error:
            if debug_display:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                disp_w, disp_h = debug_display.getWidth(), debug_display.getHeight()
                frame_resized = cv2.resize(frame_rgb, (disp_w, disp_h))
                ir = debug_display.imageNew(frame_resized.tobytes(), Display.RGB, disp_w, disp_h)
                debug_display.imagePaste(ir, 0, 0, False)
                debug_display.imageDelete(ir) 
            else:
                print("[WARN] OpenCV GUI not supported & No Webots Display found. Disabling DEBUG_VIEW.")
                DEBUG_VIEW = False
        
    return det

# MAIN LOOP

arm.reset()
gripper.open()

# Tracking list for collected bottles
collected_ids = set()
find_closest_bottle(robot.getSelf(), collected_ids)

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
            
            # Updating tracking info
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
                    base.omega = 0.6 
                    base.vx = 0; base.vy = 0
            
            elif state == "APPROACH":
                final_y = cam_h * FINAL_STOP_Y_RATIO
                ideal_y = cam_h * IDEAL_STOP_Y_RATIO
                
                if detection:
                    if center_y >= final_y:
                        base.stop()
                        state = "PICKING"
                        pickup_timer = 0
                        print(f"[STATE] PICKING REACHED")
                    else:
                        base.omega = np.clip(-error * KP_TURN, -1.0, 1.0)
                        
                        if center_y < ideal_y:
                            ratio = (ideal_y - center_y) / ideal_y
                            base.vx = np.clip(ratio * MAX_FORWARD_SPEED, STOPPING_SPEED, MAX_FORWARD_SPEED)
                        else:
                            base.vx = STOPPING_SPEED
                        base.vy = 0
                else:
                    if last_known_area > 20000:
                        print(f"[STATE] Blind Pickup Triggered.")
                        base.stop()
                        state = "PICKING"
                        pickup_timer = 0
                    elif lost_counter > LOST_DETECTION_TOLERANCE:
                        state = "SEARCHING"
                    else:
                        base.omega = np.clip(-last_known_error * KP_TURN, -0.4, 0.4)

            elif state == "PICKING":
                robot_node = robot.getSelf()
                
                # Distance limit (1.5m) for picking
                target_bottle = find_closest_bottle(robot_node, collected_ids, max_dist=1.5)
                
                if target_bottle:
                    teleport_bottle_to_back(target_bottle, robot_node, len(collected_ids))
                    b_id = target_bottle.getId()
                    collected_ids.add(b_id)
                    print(f"[ACTION] Teleported bottle {b_id} (Slot: {len(collected_ids)-1})")
                    state = "COMPLETE"
                else:
                    print("No bottle nearby (Possible ghost detection or too far). Resetting.")
                    state = "COMPLETE" 
                
                base.stop()
                pickup_timer = 0
            
            elif state == "COMPLETE":
                pickup_timer += 1
                if pickup_timer > 30: 
                    state = "SEARCHING"
                    pickup_timer = 0
                    target_lock_counter = 0
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