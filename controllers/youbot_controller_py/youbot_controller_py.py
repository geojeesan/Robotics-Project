ENABLE=True
if not ENABLE:
    exit()

from controller import Robot, Supervisor, Display
import cv2
import numpy as np
import math
import json

# ===============================================================
# IMPORTS: YOLO & PARTICLE FILTER
# ===============================================================
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from torch.nn import Module
torch.serialization.add_safe_globals([DetectionModel, Sequential, Module])
from youbot_library import ParticleFilter, ParticleVisualizer

# ===============================================================
# PARAMETERS
# ===============================================================
TIME_STEP = 32
DEBUG_VIEW = True
FRAME_SKIP = 1

# YOLO Settings
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.40                
TARGET_CLASSES = [39]          # 39 = bottle (COCO dataset)
CAMERA_RESIZE = (320, 240)     

# Motion & State
SCAN_SPEED = 1.5               
KP_TURN = 0.005                


# Localization Settings
NUM_PARTICLES = 500
MAP_FILE = "final_map.npy"

# Robot geometry
WHEEL_RADIUS = 0.05
LX = 0.228
LY = 0.158

# ===============================================================
# INITIALIZATION
# ===============================================================

robot = Supervisor()
robot_name = robot.getName()

# Sensors
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)
cam_w = camera.getWidth()
cam_h = camera.getHeight()

lidar = robot.getDevice("lidar")
lidar.enable(TIME_STEP)
lidar.enablePointCloud()       # optional but helpful
lidar_res = lidar.getHorizontalResolution()   # 512
lidar_fov = lidar.getFov()                    # 1.5 rad
lidar_max = lidar.getMaxRange()               # 8 m


compass = robot.getDevice("compass")
compass.enable(TIME_STEP)

gps = robot.getDevice("gps")
gps.enable(TIME_STEP)

emitter = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")
receiver.enable(TIME_STEP)
receiver.setChannel(1)

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
    print(f"[WARN] Failed to init PF ({e}). Falling back to raw GPS.")
    pf_active = False

# Load YOLO
print("[INFO] Loading YOLOv8 model...")
yolo = YOLO(YOLO_MODEL)
print("[INFO] YOLO Ready.")

# ===============================================================
# HELPER FUNCTIONS
# ===============================================================

def base_move(vx, vy, omega):
    global current_vx, current_vy, current_omega
    current_vx = vx
    current_vy = vy
    current_omega = omega

    speeds = [
        1 / WHEEL_RADIUS * (vx + vy + (LX + LY) * omega),
        1 / WHEEL_RADIUS * (vx - vy - (LX + LY) * omega),
        1 / WHEEL_RADIUS * (vx - vy + (LX + LY) * omega),
        1 / WHEEL_RADIUS * (vx + vy - (LX + LY) * omega)
    ]
    for i, wheel in enumerate(wheels):
        wheel.setVelocity(speeds[i])

def base_stop():
    base_move(0, 0, 0)

def get_heading():

    compass_values = compass.getValues()
    return math.atan2(compass_values[0], compass_values[1])

def lidar_distance_from_camera_pixel(cx, cam_width, lidar_ranges):
    """
    Convert camera pixel index -> lidar beam index.
    cx: camera pixel center (0..cam_width)
    lidar_ranges: list of floats, size = lidar_res (512)
    Returns distance (m) or inf.
    """
    if lidar_ranges is None or len(lidar_ranges) == 0:
        return float('inf')

    # camera pixel → normalized angle offset (-0.5 .. +0.5)
    norm = (cx / cam_width) - 0.5        # -0.5 left, +0.5 right

    # convert into lidar beam index
    # lidar beams: 0 = leftmost, (res-1) = rightmost
    lidar_index = int((norm + 0.5) * lidar_res)

    # clamp
    lidar_index = max(0, min(lidar_index, lidar_res - 1))

    return lidar_ranges[lidar_index]


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
            
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 >= cam_w:
                x2 = cam_w - 1
            if y2 >= cam_h:
                y2 = cam_h - 1
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
DEBUG=True
collected_bottles=[]

print(f"[{robot_name}] Controller Started. Waiting for localization (PF Pos + Compass Rot)...")
   

while robot.step(TIME_STEP) != -1:
    image = camera.getImage()
    if pf_active:
        dt = TIME_STEP / 1000.0        
        # A. Motion Update
        if abs(current_vx) > 0.01 or abs(current_vy) > 0.01 or abs(current_omega) > 0.01:
            pf.motion_update(current_vx, current_vy, current_omega, dt)
        
        # B. Sensor Update
        gps_vals = gps.getValues()
        pf.sensor_update_gps(gps_vals[0], gps_vals[1])
        
        lidar_ranges = lidar.getRangeImage()
        if lidar_ranges:
            pf.sensor_update_lidar(lidar_ranges, lidar_fov)
            
        # C. Resample & Estimate
        pf.resample()
        # We only use X and Y from the PF. 
        # Theta (pf_theta_est) is ignored for bottle logic.
        est_x, est_y, pf_theta_est = pf.get_estimate()
        
        # D. Visualization (still helpful to see PF internal state)
        vis.update(pf.particles, est_x, est_y, pf_theta_est, gps_vals[0], gps_vals[1])
    else:
        est_x, est_y, _ = gps.getValues()

    # -----------------------------------------------------------
    # 2. STATE MACHINE
    # -----------------------------------------------------------

    if state == "INIT":
        localization_timer += 1
        base_move(0, 0, 1)
        if localization_timer > 40:
            print(f"[{robot_name}] Localized. Starting 360 Scan.")
            print(f"PF estimate: {pf.get_estimate()}")
            print(f"GPS coords at this time for {robot_name} are: {gps.getValues()}")
            accumulated_rotation = 0.0
            state = "SCANNING"
            last_heading = get_heading() 
            base_move(0, 0, 1)

            
            
    elif state == "SCANNING":
        current_heading = get_heading()
        
        delta = current_heading - last_heading
        print(f"difference between headings: {delta}")
        if delta > math.pi: delta -= 2*math.pi
        elif delta < -math.pi: delta += 2*math.pi
        
        accumulated_rotation += abs(delta)
        print(f"Rotation so far for [{robot_name}]: {accumulated_rotation} : {((accumulated_rotation/(2*math.pi)))*100:.2f}% complete")
        last_heading = current_heading
        # -----------------------------------
        
        if image:
            detection = detect_bottle(image)
            if detection:
                # if DEBUG:
                
                    # cx, cy, _ = detection
                    # fov = camera.getFov()
                    # offset_angle = ((cx / cam_w) - 0.5) * fov
                    # print("px:", cx, "offset_deg:", math.degrees(offset_angle))
                
                    # using your current convention
                    # t1 = current_heading - offset_angle
                    # alternative convention
                    # t2 = current_heading + offset_angle
                
                    # compute two candidate world points  (use small distance 1.0m for quick test)
                    # x1 = est_x + 1.0*math.cos(t1); y1 = est_y + 1.0*math.sin(t1)
                    # x2 = est_x + 1.0*math.cos(t2); y2 = est_y + 1.0*math.sin(t2)
                
                    # print("heading_deg:", math.degrees(current_heading),
                          # "proj1_deg:", math.degrees(t1), "->", (x1,y1),
                          # "proj2_deg:", math.degrees(t2), "->", (x2,y2))
                center_x, center_y, _ = detection
                lidar_ranges = lidar.getRangeImage()
                raw_dist = lidar_distance_from_camera_pixel(center_x, cam_w, lidar_ranges)

                
                if raw_dist != float('inf') and raw_dist < 5.0:
                    fov = camera.getFov()
                    offset_angle = ((center_x / cam_w) - 0.5) * fov
                    half_fov = 0.5 * fov
                    if offset_angle > half_fov:
                        offset_angle = half_fov
                    elif offset_angle < -half_fov:
                        offset_angle = -half_fov

                    heading = get_heading()

                    # Use '+' sign because debug showed that matches your setup
                    total_angle = heading + offset_angle

                    obj_world_x = est_x + raw_dist * math.cos(total_angle)
                    obj_world_y = est_y + raw_dist * math.sin(total_angle)

                    # --------------------------------------------------
                    
                    is_new = True
                    for trash in detected_trash_list:
                        d = math.sqrt((trash['x'] - obj_world_x)**2 + (trash['y'] - obj_world_y)**2)
                        if d < 0.8: # they neey to be at least 70cm apart to be considered different (increased from 50 to acc for noise). 
                            is_new = False
                            break
                    
                    if is_new:
                        detected_trash_list.append({"type":"bottle", "x":obj_world_x, "y":obj_world_y})
                        print(f"[{robot_name}] Bottle: ({obj_world_x:.2f}, {obj_world_y:.2f})")

        if accumulated_rotation >=  0.50*2* math.pi: # no need to do a full rotation to save time. 
            base_stop()
            state = "COMPILING"
            print(f"[{robot_name}] Scan complete. Found: {len(detected_trash_list)} bottles.")

    elif state == "COMPILING":
        # for entry in detected_trash_list:
            # create_cone(entry['x'], entry['y'], 0, f"CONE_{robot_name}_{int(entry['x']*10)}")

        # Send Compass Heading as the Robot's orientation
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

    elif state == "WAITING":
        # receive messages 
         while receiver.getQueueLength() > 0:
             packet=receiver.getString()
             data=json.loads(packet)
             if robot_name in data:
                 state = "APPROACHING" # goto trash
                 goto_x = data[robot_name]["goto_x"]
                 goto_y = data[robot_name]["goto_y"]
                 curr_heading = get_heading() 
                 target_heading = math.atan2(goto_y,goto_x)
                 delta = abs(target_heading-curr_heading)
                 # rotate to face delta
                 base_move(0,0,delta)
                 state="PICKING"

             else:
                 # what to do if the robot was never assigned anything??
                 state=="SEARCHING" # go back to searching I guess?
    elif state == "PICKING":
        # we now need to get the A* path from the supervisor. 
        path = data[robot_name]["path"]
        # need a function to follow this path 
        
             

cv2.destroyAllWindows()