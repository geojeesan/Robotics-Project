ENABLE=True
if not ENABLE:
    exit()

from controller import Robot, Supervisor
import cv2
import numpy as np
import math
import json

# ===============================================================
# IMPORTS: YOLO & PARTICLE FILTER
# ===============================================================
from ultralytics import YOLO
import torch
try:
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn.modules.container import Sequential
    from torch.nn import Module
    torch.serialization.add_safe_globals([DetectionModel, Sequential, Module])
except Exception:
    pass
from youbot_library import ParticleFilter, ParticleVisualizer

# ===============================================================
# PARAMETERS
# ===============================================================
TIME_STEP = 32
DEBUG_VIEW = True

# YOLO Settings
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.40                
TARGET_CLASSES = [39]           # 39 = bottle
CAMERA_RESIZE = (320, 240)      

# Motion & State
KP_TURN = 0.005                 

# Localization Settings
NUM_PARTICLES = 500
MAP_FILE = "final_map.npy"

# Robot geometry
WHEEL_RADIUS = 0.05
LX = 0.228
LY = 0.158

# Controller gains
KP_POS = 1.2       
MAX_VX = 0.6       
MAX_VY = 0.4       
KP_ANG = 2.0       
MAX_OMEGA = 1.0    
ARRIVE_DIST = 0.15 
SLOW_DIST = 0.6    
INTERRUPT_TOLERANCE = 1.0 

# Area Threshold for Picking (approx 25% of 320x240 image)
PICKUP_AREA_THRESHOLD = 20000

# Tray Slots
TRAY_SLOTS = [
    (-0.2, -0.1), (-0.2,  0.0), (-0.2,  0.1),
    (-0.1,  0.1), (-0.1,  0.0), (-0.1, -0.1)
]

INTERRUPT_TOLERANCE = 1.0
COLLISION_DIST = 0.40

# ===============================================================
# INITIALIZATION
# ===============================================================

robot = Supervisor()
robot_name = robot.getName()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

# Sensors
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)
cam_w = camera.getWidth()
cam_h = camera.getHeight()

range_finder = robot.getDevice("range-finder")
range_finder.enable(TIME_STEP)
rf_w = range_finder.getWidth()
rf_h = range_finder.getHeight()

compass = robot.getDevice("compass")
compass.enable(TIME_STEP)

gps = robot.getDevice("gps")
gps.enable(TIME_STEP)

emitter = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")
receiver.enable(TIME_STEP)
receiver.setChannel(-1)

MAX_WHEEL_SPEED = 14.0
CAMERA_OFFSET = (0.28, 0.05, 0.0)
VISUAL_LOST_TIMEOUT = 1.0 
visual_lost_timer = 0.0

# Wheels
wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
wheels = []
for name in wheel_names:
    motor = robot.getDevice(name)
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
    wheels.append(motor)

# Velocity trackers for PF
current_vx = 0.0
current_vy = 0.0
current_omega = 0.0

# Initialize Particle Filter
print("[INFO] Initializing Particle Filter...")
try:
    pf = ParticleFilter(MAP_FILE, NUM_PARTICLES)
    vis = ParticleVisualizer(robot, pf.map_grid)
    pf_active = False
except Exception as e:
    print(f"[WARN] Failed to init PF ({e}). GPS fallback.")
    pf_active = False

# Load YOLO
print("[INFO] Loading YOLOv8 model...")
# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")
if device == 'cuda':
    print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

yolo = YOLO(YOLO_MODEL)
yolo.to(device)

# ===============================================================
# TELEPORT / PICKING LOGIC
# ===============================================================

def find_closest_bottle(robot_node, ignored_ids=None, max_dist=1.0):
    if ignored_ids is None: ignored_ids = set()
    root = robot.getRoot()
    children = root.getField("children")
    best_node = None
    min_dist_sq = float('inf')
    max_dist_sq = max_dist ** 2
    
    rx, ry, _ = robot_node.getPosition()

    for i in range(children.getCount()):
        node = children.getMFNode(i)
        if node is None or node.getId() in ignored_ids: continue
        
        n_type = node.getTypeName()
        n_def = node.getDef()
        
        is_bottle = ("WaterBottle" in n_type or "Bottle" in n_type or "BOTTLE" in n_def.upper())
        if is_bottle:
            bx, by, bz = node.getPosition()
            dist_sq = (bx - rx)**2 + (by - ry)**2
            if dist_sq < max_dist_sq and dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_node = node
    return best_node

def teleport_bottle_to_back(target_node, robot_node, slot_index=0):
    if not target_node or not robot_node: return

    rx, ry, rz = robot_node.getPosition()
    rot = robot_node.getOrientation()
    
    safe_index = slot_index % len(TRAY_SLOTS)
    off_x, off_y = TRAY_SLOTS[safe_index]
    off_z = 0.10 + (0.25 if slot_index >= len(TRAY_SLOTS) else 0)

    wx = rot[0]*off_x + rot[1]*off_y + rot[2]*off_z
    wy = rot[3]*off_x + rot[4]*off_y + rot[5]*off_z
    wz = rot[6]*off_x + rot[7]*off_y + rot[8]*off_z
    
    new_pos = [rx + wx, ry + wy, rz + wz]
    target_node.resetPhysics()
    target_node.getField("translation").setSFVec3f(new_pos)
    target_node.getField("rotation").setSFRotation([0, 0, 1, 0])

# ===============================================================
# HELPER FUNCTIONS
# ===============================================================

def normalize_angle(angle):
    while angle > math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

def base_move(vx, vy, omega):
    global current_vx, current_vy, current_omega
    current_vx = vx
    current_vy = vy
    current_omega = omega

    raw = [
        (vx + vy + (LX + LY) * omega) / WHEEL_RADIUS,
        (vx - vy - (LX + LY) * omega) / WHEEL_RADIUS,
        (vx - vy + (LX + LY) * omega) / WHEEL_RADIUS,
        (vx + vy - (LX + LY) * omega) / WHEEL_RADIUS
    ]

    max_raw = max(abs(s) for s in raw)
    if max_raw > MAX_WHEEL_SPEED:
        scale = MAX_WHEEL_SPEED / max_raw
        speeds = [s * scale for s in raw]
    else:
        speeds = raw

    for i, wheel in enumerate(wheels):
        wheel.setVelocity(speeds[i])

def base_stop():
    base_move(0, 0, 0)

def get_heading():
    rot = robot.getSelf().getOrientation()
    f_wx = rot[0]
    f_wy = rot[3]
    heading = math.atan2(f_wy, f_wx)
    return normalize_angle(heading)

def create_cone(x, y, z, name=""):
    root_node = robot.getRoot()
    r_name = robot.getName()
    robot_color = {"youbot_1":(1,0,0), "youbot_2":(0,1,0), "youbot_3":(0,0,1), "youbot_4":(1,1,0)}
    r, g, b = robot_color.get(r_name, (1,1,1))
    
    children_field = root_node.getField('children')
    cone_vrml = f"""
    DEF {name} Solid {{
      translation {x} {y} {z}
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor {r} {g} {b} roughness 1 metalness 0 transparency 0.5 }}
          geometry Cone {{ bottomRadius 0.05 height 0.3 }}
        }}
      ]
      name "{name}"
    }}
    """
    children_field.importMFNodeFromString(-1, cone_vrml)

# ===============================================================
# YOLO DETECTION
# ===============================================================
def detect_bottle(image):
    if yolo is None or image is None: return None
    
    frame = np.frombuffer(image, np.uint8).reshape((cam_h, cam_w, 4))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Pass device=device to ensure inference happens on GPU
    results = yolo(cv2.resize(frame_bgr, CAMERA_RESIZE), 
                   conf=YOLO_CONF, classes=TARGET_CLASSES, verbose=False, device=device)
    
    det = None
    best_area = 0
    boxes = results[0].boxes
    
    if boxes:
        sx, sy = cam_w / CAMERA_RESIZE[0], cam_h / CAMERA_RESIZE[1]
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1 = int(xyxy[0]*sx), int(xyxy[1]*sy)
            x2, y2 = int(xyxy[2]*sx), int(xyxy[3]*sy)
            
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(cam_w - 1, x2); y2 = min(cam_h - 1, y2)
            
            if x2 > x1 and y2 > y1:
                roi = frame_bgr[y1:y2, x1:x2]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv_roi, np.array([0, 100, 50]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv_roi, np.array([160, 100, 50]), np.array([180, 255, 255]))
                mask = cv2.bitwise_or(mask1, mask2)
                
                red_pixels = cv2.countNonZero(mask)
                total_pixels = (x2-x1) * (y2-y1)
                red_ratio = red_pixels / total_pixels
                
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
                
                if DEBUG_VIEW:
                    cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,255,0), 2)

    if DEBUG_VIEW:
        cv2.imshow(f"YOLO {robot_name}", frame_bgr)
        cv2.waitKey(1)
        
    return det


# ===============================================================
# MAIN LOOP
# ===============================================================

detected_trash_list = [] 
accumulated_rotation = 0.0
last_heading = 0.0
state = "INIT"
localization_timer = 0
collected_bottles = set()
current_waypoint_index = 0
path = []
final_goal_world = (0,0)
ROTATE_ONLY_TIMEOUT = 2.0
ROTATION_FORWARD_SPEED = 0.12
rotation_only_timer = 0.0

print(f"[{robot_name}] Controller Started.")
    
while robot.step(TIME_STEP) != -1:
    image = camera.getImage()
    

    # PF Logic
    if pf_active:
        dt = TIME_STEP / 1000.0        
        if abs(current_vx) > 0.01 or abs(current_vy) > 0.01 or abs(current_omega) > 0.01:
            pf.motion_update(current_vx, current_vy, current_omega, dt)
        gps_vals = gps.getValues()
        pf.sensor_update_gps(gps_vals[0], gps_vals[1])
        pf.resample()
        est_x, est_y, pf_theta_est = pf.get_estimate()
        vis.update(pf.particles, est_x, est_y, pf_theta_est, gps_vals[0], gps_vals[1])
    else:
        est_x, est_y, _ = gps.getValues()
        # --- NEW: PASSIVE COLLISION COLLECTION ---
    # Check if we are touching any bottle (independent of current State)
    touching_node = find_closest_bottle(robot.getSelf(), collected_bottles, max_dist=COLLISION_DIST)
    
    if touching_node:
        bottle_id = touching_node.getId()
        print(f"[{robot_name}] COLLISION DETECTED! collecting bottle ID {bottle_id}")
        
        # 1. Teleport Bottle
        teleport_bottle_to_back(touching_node, robot.getSelf(), len(collected_bottles))
        collected_bottles.add(bottle_id)
        
        # 2. Remove from internal memory (so we don't try to go to it later)
        # We filter the list to keep only trashes that are NOT close to the collected one
        # (This prevents ghost targets)
        bx, by, _ = touching_node.getPosition()
        detected_trash_list = [
            t for t in detected_trash_list 
            if math.hypot(t['x'] - bx, t['y'] - by) > 0.6
        ]

        # 3. Notify Supervisor (Update global map)
        current_heading = get_heading()
        payload = {
            "robot": robot_name,
            "pose": {"x": est_x, "y": est_y, "theta": current_heading},
            "trash_world_coords": [], # No new trash seen, just status update
            "status": "WAITING" 
        }
        
        # Note: We send this to ensure the supervisor knows we are still alive/active
        emitter.send(json.dumps(payload).encode('utf-8'))
        state="WAITING"

    # -----------------------------------------------------------
    # 2. STATE MACHINE
    # -----------------------------------------------------------

    if state == "INIT":
        localization_timer += 1
        base_move(0, 0, 0) 
        
        if localization_timer > 40:
            print(f"[{robot_name}] Localized. Starting 360 Scan.")
            accumulated_rotation = 0.0
            state = "SCANNING"
            last_heading = get_heading() 
            base_move(0, 0, 1) 


    elif state == "SCANNING":
    
        current_heading = get_heading()
        delta = current_heading - last_heading
        
        # Heading wrap-around handling
        if delta > math.pi: delta -= 2*math.pi
        elif delta < -math.pi: delta += 2*math.pi
        
        # Accumulate absolute rotation
        accumulated_rotation += abs(delta)
        last_heading = current_heading
        
        # Debugging Scan Progress
        if robot.getTime() % 0.5 < 0.05:
             print(f"[{robot_name}] Scan: {accumulated_rotation:.2f} / 6.28")

        # Detection Logic
        if image:
            detection = detect_bottle(image)
            if detection:
                center_x, center_y, _ = detection
                
                rf_data = range_finder.getRangeImage()
                idx_x = min(max(0, center_x), rf_w - 1)
                idx_y = min(max(0, center_y), rf_h - 1)
                rf_index = int(idx_y * rf_w + idx_x)
                raw_dist = rf_data[rf_index]
                
                if raw_dist != float('inf') and raw_dist < 3:
                    fov = camera.getFov()
                    # Left pixel (0) -> Positive Angle (Left Turn)
                    offset_angle = -1 * ((center_x / cam_w) - 0.5) * fov
                    
                    heading = get_heading()
                    total_angle = heading + offset_angle
                    
                    rot = robot.getSelf().getOrientation()
                    R00, R01 = rot[0], rot[1]
                    R10, R11 = rot[3], rot[4]
                    cam_dx, cam_dy, _ = CAMERA_OFFSET
                    
                    sensor_world_x = est_x + (R00 * cam_dx + R01 * cam_dy)
                    sensor_world_y = est_y + (R10 * cam_dx + R11 * cam_dy)
                    obj_world_x = sensor_world_x + raw_dist * math.cos(total_angle)
                    obj_world_y = sensor_world_y + raw_dist * math.sin(total_angle)

                    is_new = True
                    for trash in detected_trash_list:
                        d = math.sqrt((trash['x'] - obj_world_x)**2 + (trash['y'] - obj_world_y)**2)
                        if d < 0.6: 
                            is_new = False
                            break
                    
                    if is_new:
                        detected_trash_list.append({"type":"bottle", "x":obj_world_x, "y":obj_world_y})
                        print(f"[{robot_name}] Found Bottle: ({obj_world_x:.2f}, {obj_world_y:.2f})")

        if accumulated_rotation >= 2* math.pi: 
            base_stop()
            state = "COMPILING"
            print(f"[{robot_name}] Scan complete. Found: {len(detected_trash_list)} bottles.")

    elif state == "COMPILING":
        current_heading = get_heading()
        payload = {
            "robot": robot_name,
            "pose": {"x": est_x, "y": est_y, "theta": current_heading},
            "trash_world_coords": detected_trash_list,
            "status": "WAITING"
        }
        msg = json.dumps(payload)
        emitter.send(msg.encode('utf-8'))
        print(f"[{robot_name}] Data sent. Status: WAITING")
        state = "WAITING"
        print(state)

    elif state == "WAITING":
        base_stop()
        
        while receiver.getQueueLength() > 0:
            packet = receiver.getString()
            receiver.nextPacket() 
            try:
                data = json.loads(packet)
                if robot_name in data:
                    print(f"[{robot_name}] Task received!")
                    raw_path = data[robot_name].get("path", ())
                    
                    if len(raw_path) > 0:
                        path = raw_path
                        current_waypoint_index = 0
                        final_goal_world = (path[-1][0], path[-1][1])
                        state = "APPROACHING"
                        print(f"[{robot_name}] {state}")
                        print(f"[{robot_name}] Following path with {len(path)} waypoints. Goal: {final_goal_world}")
                    else:
                        print(f"[{robot_name}] Received empty path.")
            except ValueError:
                pass

    elif state == "APPROACHING":
    # Standard A* Path Following
        if path and current_waypoint_index < len(path):
            target = path[current_waypoint_index]
            target_x, target_y = float(target[0]), float(target[1])
    
            theta = get_heading()
            dx = target_x - est_x
            dy = target_y - est_y
            distance = math.hypot(dx, dy)
            target_angle = math.atan2(dy, dx)
            heading_error = normalize_angle(target_angle - theta)
    
            # --- arrival / waypoint advance ---
            if distance < ARRIVE_DIST:   # use ARRIVE_DIST constant you set earlier (0.15)
                current_waypoint_index += 3
                if current_waypoint_index >= len(path):
                    print(f"[{robot_name}] Reached end of waypoints (index {current_waypoint_index} / {len(path)}). Switching to Visual Servoing.")
                    base_stop()
                    state = "VISUAL_SERVOING"
                    rotation_only_timer = 0.0
                    continue
                rotation_only_timer=0.0
                continue
                
            vx_cmd = 0.0
            omega_cmd = 0.0
    
            # When heading error large: normally rotate-only
            if abs(heading_error) > 0.15:
                omega_cmd = KP_ANG * heading_error
                vx_cmd = 0.0
    
                # rotation-only timer update (TIME_STEP is in ms)
                rotation_only_timer += (TIME_STEP / 1000.0)
    
                # if we've been rotating-only too long, try a recovery:
                if rotation_only_timer >= ROTATE_ONLY_TIMEOUT:
                    print(f"[{robot_name}] Rotation watchdog triggered (headed {heading_error:.2f} rad). attempting recovery.")
                    vx_cmd = ROTATION_FORWARD_SPEED
                    rotation_only_timer = 0.0
    
            else:
                # heading aligned enough: drive toward target normally
                omega_cmd = KP_ANG * heading_error
                vx_cmd = KP_POS * distance
                # reset rotation-only timer when we are making forward progress
                rotation_only_timer = 0.0
    
            # saturate and slow when close
            vx_cmd = max(min(vx_cmd, MAX_VX), -MAX_VX)
            if distance < SLOW_DIST:
                vx_cmd *= (distance / SLOW_DIST)
            omega_cmd = max(min(omega_cmd, MAX_OMEGA), -MAX_OMEGA)
    
            base_move(vx_cmd, 0.0, omega_cmd)
    
        else:
            # End of Path -> Try to Pickup
            print(f"[{robot_name}] Path Complete. Switching to Visual Servoing.")
            base_stop()
            state = "VISUAL_SERVOING"
    
            print("[{robot_name}] {state}")

    elif state == "VISUAL_SERVOING":
        if image:
            detection = detect_bottle(image)
            
            if detection:
                # We have a visual target
                visual_lost_timer=0
                cx, cy, area = detection
                
                # Check AREA threshold for picking (Close Enough?)
                if area > PICKUP_AREA_THRESHOLD:
                    print(f"[{robot_name}] Bottle Close Enough (Area {area}). Picking.")
                    base_stop()
                    state = "PICKING"
                    print(f"[{robot_name}] {state}")
                else:
                    err_x = (cx / cam_w) - 0.5
                    omega = -err_x * 2.0
                    vx = 0.7  # Slow approach speed
                    base_move(vx, 0, omega)
            else:
                visual_lost_timer+=dt
                
                if visual_lost_timer >= VISUAL_LOST_TIMEOUT:

                    print(f"[{robot_name}] Searching for target (Spinning)...")
                    visual_lost_timer=0                 
                    accumulated_rotation = 0.0
                    last_heading=get_heading()
                    base_move(0, 0, 1) 
                    state="SCANNING"

    elif state == "PICKING":
        print(f"[{robot_name}] Attempting to teleport/pick bottle...")
        target_node = find_closest_bottle(robot.getSelf(), collected_bottles, max_dist=1.0)
        
        if target_node:
            teleport_bottle_to_back(target_node, robot.getSelf(), len(collected_bottles))
            collected_bottles.add(target_node.getId())
            print(f"[{robot_name}] SUCCESS: Collected bottle ID {target_node.getId()}")
            
            current_heading = get_heading()
            payload = {
                "robot": robot_name,
                "pose": {"x": est_x, "y": est_y, "theta": current_heading},
                "trash_world_coords": [],
                "status": "WAITING" 
            }
            emitter.send(json.dumps(payload).encode('utf-8'))
        else:
            print(f"[{robot_name}] FAILED: No physical bottle found.")
            current_heading = get_heading()
            payload = {
                "robot": robot_name,
                "pose": {"x": est_x, "y": est_y, "theta": current_heading},
                "trash_world_coords": [],
                "status": "WAITING"
            }
            emitter.send(json.dumps(payload).encode('utf-8'))

        for _ in range(10): robot.step(TIME_STEP)
        state = "WAITING"

cv2.destroyAllWindows()