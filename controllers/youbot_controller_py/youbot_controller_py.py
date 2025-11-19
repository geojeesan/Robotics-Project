from controller import Robot
import cv2
import numpy as np
# [MCL] Import the particle filter class
from particle_filter import ParticleFilter

# ===============================================================
# PARAMETERS
# ===============================================================
TIME_STEP = 64
DEBUG_VIEW = False
CAMERA_WARMUP_STEPS = 5
MIN_AREA = 100
TARGET_AREA = 15000
STOPPING_SPEED = 0.05
FRAME_SKIP = 1
TARGET_LOCK_THRESHOLD = 10
MAX_AREA_CHANGE_RATIO = 5.0
MAX_POSITION_CHANGE = 150
LOST_DETECTION_TOLERANCE = 50
APPROACH_AREA_MULTIPLIER = 1.8
APPROACH_CONFIRMATION_FRAMES = 8
IDEAL_STOP_Y_RATIO = 0.90
FINAL_STOP_Y_RATIO = 0.95

# Robot geometry for omnidirectional base
WHEEL_RADIUS = 0.05
LX = 0.228
LY = 0.158

# [MCL] Parameters for Monte Carlo Localization
NUM_PARTICLES = 500          # Number of particles (REDUCED from 1000)
PF_CONVERGENCE_STD = 0.2    # meters (std dev to be considered 'converged')
WORLD_SIZE_M = 10.0         # Assume a 10m x 10m world
MAP_RESOLUTION_M = 0.1      # 10cm grid cells
MCL_UPDATE_SKIP = 10        # [MCL] NEW: Run the slow update every 10 steps

# [MCL] NEW: Parameters for the Lidar Sensor Model
LIDAR_STD_DEV_M = 0.5       # Standard deviation for lidar noise (in meters). TUNE THIS.
LIDAR_DOWNSAMPLE = 20       # Check only every 20th ray to speed up. (INCREASED from 10)
LIDAR_MAX_RANGE = 0.0       # Will be set from the lidar device
LIDAR_FOV = 0.0             # Will be set from the lidar device
LIDAR_NUM_RAYS = 0          # Will be set from the lidar device
LIDAR_ANGLES = np.array([]) # Will be set from the lidar device
MAP_SIZE_PIXELS = 0         # Will be set after map creation

# ===============================================================
# INITIALIZATION
# ===============================================================
robot = Robot()
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

# [MCL] Initialize LiDAR
# !!! YOU MUST ADD A 'Lidar' NODE NAMED 'lidar' TO YOUR ROBOT IN WEBOTS !!!
try:
    lidar = robot.getDevice("lidar")
    lidar.enable(TIME_STEP)
    lidar.enablePointCloud()
    has_lidar = True
    
    # [MCL] NEW: Get Lidar properties
    LIDAR_MAX_RANGE = lidar.getMaxRange()
    LIDAR_FOV = lidar.getFov()
    LIDAR_NUM_RAYS = lidar.getHorizontalResolution()
    # Create an array of angles for each ray
    LIDAR_ANGLES = np.linspace(-LIDAR_FOV / 2.0, LIDAR_FOV / 2.0, LIDAR_NUM_RAYS)
    
    print("[MCL] LiDAR enabled.")
    print(f"[MCL]   Max Range: {LIDAR_MAX_RANGE:.2f}m")
    print(f"[MCL]   FOV: {np.degrees(LIDAR_FOV):.1f}deg")
    print(f"[MCL]   Num Rays: {LIDAR_NUM_RAYS}")
    
except Exception as e:
    has_lidar = False
    print(f"[MCL] WARNING: Could not initialize LiDAR. Localization will not work.")
    print(f"[MCL] Error: {e}")

# [MCL] Initialize GPS and Compass (for ground truth debugging)
# !!! OPTIONAL: ADD 'GPS' AND 'Compass' NODES TO YOUR ROBOT IN WEBOTS !!!
try:
    gps = robot.getDevice("gps")
    gps.enable(TIME_STEP)
    compass = robot.getDevice("compass")
    compass.enable(TIME_STEP)
    has_ground_truth = True
    print("[MCL] GPS and Compass enabled for ground truth.")
except Exception as e:
    has_ground_truth = False
    print(f"[MCL] Note: GPS/Compass not found. Ground truth will not be available.")


# Initialize YouBot wheels
wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
wheels = []
for name in wheel_names:
    motor = robot.getDevice(name)
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
    wheels.append(motor)

# Initialize arm joints
arm_joints = []
arm_names = ["arm1", "arm2", "arm3", "arm4", "arm5"]
for name in arm_names:
    joint = robot.getDevice(name)
    joint.setVelocity(0.5)
    arm_joints.append(joint)

# Initialize gripper
try:
    gripper = robot.getDevice("finger::left")
    gripper.setVelocity(0.03)
    has_gripper = True
except:
    has_gripper = False
    gripper = None

# ===============================================================
# [MCL] HELPER FUNCTIONS (MAP, SENSOR MODEL, MOTION)
# ===============================================================

# [MCL] Global variable to store the last commanded velocity for the PF
commanded_velocity = [0.0, 0.0, 0.0]  # [vx, vy, omega]

def load_map(map_file):
    """
    Loads the occupancy grid map from a .npy file.
    """
    global MAP_SIZE_PIXELS # [MCL] NEW: Store map size
    try:
        map_data = np.load(map_file)
        MAP_SIZE_PIXELS = map_data.shape[0]
        print(f"[MCL] Successfully loaded map '{map_file}'")
        print(f"[MCL] Map size: {MAP_SIZE_PIXELS}x{MAP_SIZE_PIXELS} pixels.")
        return map_data
    except Exception as e:
        print(f"[MCL] FATAL ERROR: Could not load map '{map_file}'")
        print(f"[MCL] Did you run the 'mapper.py' controller first?")
        print(f"[MCL] Error: {e}")
        return None

def create_dummy_map(world_size_m, resolution_m):
    """
    Creates a simple dummy map (e.g., a 10x10m box).
    1 = occupied (wall), 0 = free space
    """
    # [MCL] FATAL ERROR: This is the source of your pose errors.
    # The Particle Filter is comparing Lidar data from your *real* world
    # to this *empty box*. The two do not match, so the filter converges
    # to a completely random, incorrect pose.
    #
    # YOU MUST REPLACE THIS FUNCTION with one that loads a
    # real map of your Webots world (e.g., from a PNG or NumPy file).
    #
    print("\n" + "*"*60)
    print("[MCL] FATAL ERROR: Using DUMMY MAP for localization.")
    print("[MCL] The robot's estimated pose will be WRONG.")
    print("[MCL] You MUST replace 'create_dummy_map' with a real map loader.")
    print("*"*60 + "\n")
    
    global MAP_SIZE_PIXELS # [MCL] NEW: Store map size
    map_size_pixels = int(world_size_m / resolution_m)
    MAP_SIZE_PIXELS = map_size_pixels # [MCL] NEW
    map_data = np.zeros((map_size_pixels, map_size_pixels), dtype=np.uint8)
    
    # Create a border wall
    map_data[0, :] = 1
    map_data[-1, :] = 1
    map_data[:, 0] = 1
    map_data[:, -1] = 1
    
    print(f"[MCL] Created dummy map of size {map_size_pixels}x{map_size_pixels} pixels.")
    return map_data

# [MCL] NEW: Helper function to convert world (m) to map (pixels)
def world_to_map(x_m, y_m):
    """Converts world coordinates (meters) to map grid coordinates (pixels)."""
    px = int((x_m - MAP_ORIGIN_M) / MAP_RESOLUTION_M)
    # Map Y is inverted relative to world Y
    py = int((-y_m - MAP_ORIGIN_M) / MAP_RESOLUTION_M) 
    return px, py

# [MCL] NEW: Helper function to cast a single ray on the map
def _cast_ray(pose_x, pose_y, pose_theta, ray_angle, map_data):
    """Casts a single ray from a pose to find the expected distance."""
    
    # Calculate the ray's absolute angle in the world
    world_angle = pose_theta + ray_angle
    
    # Step along the ray path
    step_size = MAP_RESOLUTION_M
    current_dist = step_size
    
    while current_dist < LIDAR_MAX_RANGE:
        # Get the world coordinate point to check
        check_x = pose_x + current_dist * np.cos(world_angle)
        check_y = pose_y + current_dist * np.sin(world_angle)
        
        # Convert to map pixel coordinates
        px, py = world_to_map(check_x, check_y)
        
        # Check if out of map bounds
        if not (0 <= px < MAP_SIZE_PIXELS and 0 <= py < MAP_SIZE_PIXELS):
            return current_dist # Or LIDAR_MAX_RANGE
            
        # Check if we hit a wall (1)
        if map_data[py, px] == 1:
            return current_dist
            
        current_dist += step_size
        
    # No wall hit, return max range
    return LIDAR_MAX_RANGE


def mcl_sensor_model(lidar_scan_full, pose, map_data, map_resolution, map_origin_m):
    """
    Calculates the likelihood of a lidar scan given a pose and a map.
    p(z | x, m)
    
    [MCL] This is the new, functional sensor model.
    It will be SLOW in pure Python, so we use LIDAR_DOWNSAMPLE.
    """
    global MAP_RESOLUTION_M, MAP_ORIGIN_M, LIDAR_ANGLES, LIDAR_DOWNSAMPLE
    
    # [MCL] FIX: Slice the full scan to get only the first layer
    # This matches the fix from the GPS controller to prevent the IndexError
    # and to ensure we are only using one layer of Lidar data.
    lidar_scan = np.array(lidar_scan_full)[0:LIDAR_NUM_RAYS]
    
    # We will compute the likelihood in log-space to avoid numerical underflow
    total_log_likelihood = 0.0
    
    # Get the particle's pose
    pose_x, pose_y, pose_theta = pose
    
    # Loop through a downsampled set of rays to save computation
    for i in range(0, LIDAR_NUM_RAYS, LIDAR_DOWNSAMPLE):
        z_actual = lidar_scan[i]
        
        # Handle 'inf' values from Lidar
        if z_actual > LIDAR_MAX_RANGE - 0.1 or np.isinf(z_actual):
            z_actual = LIDAR_MAX_RANGE
            
        # Get the angle of this specific ray
        ray_angle = LIDAR_ANGLES[i]
        
        # 1. CALCULATE EXPECTED DISTANCE (z_expected)
        # Cast a ray from the particle's pose on the map
        z_expected = _cast_ray(pose_x, pose_y, pose_theta, ray_angle, map_data)
        
        # 2. COMPARE z_actual to z_expected
        # We use a Gaussian (Normal) distribution to score the difference.
        # p(z_actual | z_expected) = exp( - (z_actual - z_expected)^2 / (2 * sigma^2) )
        # We use the log of this: - (diff^2) / (2 * sigma^2)
        
        diff = z_actual - z_expected
        log_prob = - (diff ** 2) / (2 * LIDAR_STD_DEV_M ** 2)
        
        total_log_likelihood += log_prob

    # Convert back from log-space to a probability (likelihood)
    # Clamp the log likelihood to prevent np.exp from overflowing/underflowing
    # This is a common practical trick.
    clamped_log_likelihood = np.clip(total_log_likelihood, -50.0, 0.0)
    
    return np.exp(clamped_log_likelihood)


# [MCL] Initialize the map and the particle filter
# [MCL] NEW: Store map origin globally
MAP_ORIGIN_M = -WORLD_SIZE_M / 2.0 
MAP_DATA = load_map('map.npy') # [MCL] MODIFIED: Load the real map!

# [MCL] Exit if map failed to load
if MAP_DATA is None:
    exit()

pf = ParticleFilter(
    num_particles=NUM_PARTICLES,
    map_data=MAP_DATA,
    sensor_model_func=mcl_sensor_model, # Pass the corrected function
    map_resolution=MAP_RESOLUTION_M,
    world_size_m=WORLD_SIZE_M
)

# ===============================================================
# HELPER FUNCTIONS (ROBOT)
# ===============================================================
def base_move(vx, vy, omega):
    """
    Move the omnidirectional base.
    [MCL] This function is modified to store the commanded velocity.
    """
    global commanded_velocity
    commanded_velocity = [vx, vy, omega] # Store for the particle filter
    
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

# (Arm and Gripper functions are unchanged)
def arm_reset():
    arm_joints[0].setPosition(0.0)
    arm_joints[1].setPosition(1.57)
    arm_joints[2].setPosition(-2.635)
    arm_joints[3].setPosition(1.78)
    arm_joints[4].setPosition(0.0)

def arm_pick_position():
    arm_joints[0].setPosition(0.0)
    arm_joints[1].setPosition(-0.97)
    arm_joints[2].setPosition(-1.55)
    arm_joints[3].setPosition(-0.61)
    arm_joints[4].setPosition(0.0)

def arm_lift_position():
    arm_joints[0].setPosition(2.949)
    arm_joints[1].setPosition(0.92)
    arm_joints[2].setPosition(0.42)
    arm_joints[3].setPosition(1.78)
    arm_joints[4].setPosition(0.0)

def gripper_open():
    if has_gripper: gripper.setPosition(0.025)

def gripper_close():
    if has_gripper: gripper.setPosition(0.0)

# ===============================================================
# DETECT RED OBJECT (Unchanged)
# ===============================================================
def detect_red_object(image, target_position=None, target_area=None):
    img = np.frombuffer(image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            valid_objects.append((center_x, center_y, area, x, y, w, h))
    detected = None
    tracked_obj = None
    img_center_x = camera.getWidth() // 2
    img_center_y = camera.getHeight() // 2
    img_width = camera.getWidth()
    img_height = camera.getHeight()
    is_locked = target_position is not None and target_area is not None
    TRACKING_DISTANCE_THRESHOLD = 180 if is_locked else 120
    if valid_objects:
        if target_position is not None:
            target_x, target_y = target_position
            best_obj = None
            min_distance = float('inf')
            for obj in valid_objects:
                center_x, center_y = obj[0], obj[1]
                distance = np.sqrt((center_x - target_x)**2 + (center_y - target_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    best_obj = obj
            max_tracking_distance = TRACKING_DISTANCE_THRESHOLD
            if best_obj and min_distance < max_tracking_distance:
                obj_area = best_obj[2]
                area_valid = True
                if target_area is not None and target_area > 0 and obj_area < target_area * 10:
                    area_ratio = max(obj_area / target_area, target_area / obj_area)
                    if is_locked:
                        if obj_area < target_area * 0.4:
                            area_valid = False
                    else:
                        if area_ratio > MAX_AREA_CHANGE_RATIO:
                            area_valid = False
                if area_valid:
                    detected = (best_obj[0], best_obj[1], best_obj[2])
                    tracked_obj = best_obj
                else:
                    tracked_obj = None
            else:
                tracked_obj = None
        else:
            best_obj = None
            min_dist_to_center = float('inf')
            for obj in valid_objects:
                center_x, center_y = obj[0], obj[1]
                dist_to_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                if dist_to_center < min_dist_to_center:
                    min_dist_to_center = dist_to_center
                    best_obj = obj
            if best_obj:
                detected = (best_obj[0], best_obj[1], best_obj[2])
                tracked_obj = best_obj
    if DEBUG_VIEW:
        pass
    return detected

# ===============================================================
# MAIN LOOP
# ===============================================================

# Initialize arm and gripper
arm_reset()
gripper_open()

frame_count = 0
lost_counter = 0
# [MCL] Set initial state to LOCALIZING
state = "LOCALIZING"
pickup_timer = 0
approach_start_area = 0
target_position = None
target_lock_counter = 0
last_known_position = None
last_known_error = 0
last_known_area = 0
forward_direction = 1
area_trend_counter = 0
last_area_for_motion = 0
close_confirmation_counter = 0

estimated_pose = np.array([0.0, 0.0, 0.0])
dt = TIME_STEP / 1000.0

while robot.step(TIME_STEP) != -1:
    frame_count += 1
    if frame_count <= CAMERA_WARMUP_STEPS:
        continue # Wait for sensors to warm up
        
    # =================================
    # [MCL] PARTICLE FILTER UPDATE
    # =================================
    
    # 1. PREDICT: Move particles based on last command (FAST - run every step)
    pf.predict(u=commanded_velocity, dt=dt)
    
    # [MCL] NEW: Only run the expensive update/resample every N steps
    if frame_count % MCL_UPDATE_SKIP == 0:
        
        # 2. UPDATE: Get sensor readings and update weights (SLOW)
        lidar_scan = None
        if has_lidar:
            lidar_scan = lidar.getRangeImage() # Get sensor reading
            
        pf.update(z=lidar_scan) # Update weights
        
        # 3. RESAMPLE: Create new particle set (SLOW)
        pf.resample()
        
        # 4. GET ESTIMATE: Get the new best guess of our pose (FAST)
        estimated_pose = pf.get_estimated_pose()
    
    # [MCL] Print pose estimate and ground truth (if available)
    if frame_count % 50 == 0:
        print(f"[MCL] Est. Pose: x={estimated_pose[0]:.2f}m, y={estimated_pose[1]:.2f}m, th={np.degrees(estimated_pose[2]):.1f}deg")
        if has_ground_truth:
            gps_val = gps.getValues()
            compass_val = compass.getValues()
            # Convert compass (North vector) to angle
            gt_theta = np.arctan2(compass_val[0], compass_val[2])
            print(f"[MCL] Gnd. Truth: x={gps_val[0]:.2f}m, y={gps_val[2]:.2f}m, th={np.degrees(gt_theta):.1f}deg")

    # =================================
    # ROBOT STATE MACHINE
    # =================================
    
    if frame_count % FRAME_SKIP != 0:
        continue

    image = camera.getImage()
    if not image:
        continue

    is_locked = target_lock_counter > TARGET_LOCK_THRESHOLD
    detection = detect_red_object(image,
                                 target_position if is_locked else None,
                                 approach_start_area if is_locked and approach_start_area > 0 else None)

    # [MCL] BUG FIX: The PICKING/COMPLETE logic must be *outside* the
    # 'if detection:' block, otherwise the arm blocks the camera
    # and the pickup_timer stops.
    
    if state == "PICKING":
        pickup_timer += 1
        if pickup_timer < 60:
            arm_pick_position()
            gripper_open()
        elif pickup_timer < 100:
            gripper_close()
        elif pickup_timer < 180:
            arm_lift_position()
        else:
            state = "COMPLETE"
            base_stop() # Explicitly stop here

    elif state == "COMPLETE":
        pickup_timer += 1
        if pickup_timer == 1:
            arm_reset()
            gripper_open()
            target_position = None
            target_lock_counter = 0
            # ... (rest of reset variables) ...
            lost_counter = 0
        elif pickup_timer > 30:
            state = "SEARCHING"
            pickup_timer = 0
        else:
            base_stop() # Stay stopped
            
    # --- Navigation states only run if we have a detection ---
    elif detection:
        center_x, center_y, area = detection
        width = camera.getWidth()
        height = camera.getHeight()
        error = center_x - width // 2
        lost_counter = 0
        
        # (Tracking/Locking logic is unchanged)
        if target_position is None:
            target_position = (center_x, center_y)
            target_lock_counter = 1
        else:
            dist_to_target = np.sqrt((center_x - target_position[0])**2 + (center_y - target_position[1])**2)
            area_change_ok = True
            area_ratio = 1.0
            if approach_start_area > 0:
                area_ratio = max(area / approach_start_area, approach_start_area / area)
                area_change_ok = area_ratio <= MAX_AREA_CHANGE_RATIO
            if target_lock_counter <= TARGET_LOCK_THRESHOLD:
                if dist_to_target < MAX_POSITION_CHANGE and area_change_ok:
                    target_position = (center_x, center_y)
                    target_lock_counter += 1
            else:
                if dist_to_target < MAX_POSITION_CHANGE and area_change_ok:
                    target_position = (center_x, center_y)
                    target_lock_counter += 1
        
        # [MCL] New state
        if state == "LOCALIZING":
            # We are trying to localize, but we see the object.
            # We can either ignore it, or pause localization.
            # For now, we continue localizing.
            if frame_count % 50 == 0:
                print("[STATE] Localizing... (Object Spotted)")
            
            # Check for convergence
            if pf.is_converged(PF_CONVERGENCE_STD):
                print(f"[STATE] Localization converged! Pose: {estimated_pose[0]:.2f}, {estimated_pose[1]:.2f}")
                base_stop()
                state = "SEARCHING"
            else:
                # Keep spinning to get more sensor data
                base_turn(0.3)

        elif state == "SEARCHING":
            if target_lock_counter > TARGET_LOCK_THRESHOLD:
                state = "APPROACH"
                approach_start_area = area
                last_known_position = (center_x, center_y)
                last_known_error = error
                last_known_area = area
                forward_direction = 1
                close_confirmation_counter = 0
            else:
                if abs(error) > 20:
                    turn_speed = np.clip(-error / 40.0, -0.4, 0.4)
                    base_turn(turn_speed)
                else:
                    base_stop()

        elif state == "APPROACH":
            # (Approach logic is unchanged)
            last_known_position = (center_x, center_y)
            last_known_error = error
            last_known_area = area
            lost_counter = 0
            required_area = TARGET_AREA
            if approach_start_area > 0:
                required_area = max(TARGET_AREA, approach_start_area * APPROACH_AREA_MULTIPLIER)
            ideal_y_stop = height * IDEAL_STOP_Y_RATIO
            final_y_stop = height * FINAL_STOP_Y_RATIO
            if center_y > final_y_stop:
                base_stop()
                print(f"[STATE] 🛑 EMERGENCY STOP: Object center_y ({center_y:.0f}) passed final y stop ({final_y_stop:.0f}). Starting pickup sequence.")
                state = "PICKING"
                pickup_timer = 0
                lost_counter = 0
                continue
            if center_y < ideal_y_stop and required_area > approach_start_area:
                speed_ratio = (required_area - area) / (required_area - approach_start_area)
                speed_ratio = np.clip(speed_ratio, 0.0, 1.0)
                forward_speed = (0.15 - STOPPING_SPEED) * speed_ratio + STOPPING_SPEED
            else:
                forward_speed = STOPPING_SPEED * 0.5
            forward_speed *= forward_direction
            is_at_ideal_y = center_y >= ideal_y_stop
            if abs(error) > 15:
                turn_speed = np.clip(-error / 30.0, -0.5, 0.5)
                base_turn(turn_speed)
                close_confirmation_counter = 0
            elif not is_at_ideal_y:
                base_forward(forward_speed)
                close_confirmation_counter = 0
                if frame_count % 50 == 0:
                    print(f"[APPROACH] Moving toward target... speed: {forward_speed:.2f}m/s, center_y: {center_y:.0f} (ideal: {ideal_y_stop:.0f}), area: {area:.0f}")
            else:
                close_confirmation_counter += 1
                if close_confirmation_counter >= APPROACH_CONFIRMATION_FRAMES:
                    base_stop()
                    print(f"[STATE] Trash reached! Center Y: {center_y:.0f}. Starting pickup sequence...")
                    state = "PICKING"
                    pickup_timer = 0
                    lost_counter = 0
                else:
                    base_forward(STOPPING_SPEED * 0.5 * forward_direction)
                    if frame_count % 20 == 0:
                        print(f"[APPROACH] Confirming proximity ({close_confirmation_counter}/{APPROACH_CONFIRMATION_FRAMES}) - center_y: {center_y:.0f}")

        # [MCL] BUG FIX: Removed PICKING and COMPLETE states from here.
        # elif state == "PICKING": ...
        # elif state == "COMPLETE": ...
            
    else:
        # Lost Detection Logic
        lost_counter += 1

        # [MCL] New state
        if state == "LOCALIZING":
            # We are spinning to localize, just keep spinning.
            base_turn(0.3)
            if frame_count % 50 == 0:
                print("[STATE] Localizing... (spinning)")
            
            # Check for convergence
            if pf.is_converged(PF_CONVERGENCE_STD):
                print(f"[STATE] Localization converged! Pose: {estimated_pose[0]:.2f}, {estimated_pose[1]:.2f}")
                base_stop()
                state = "SEARCHING"

        elif state == "SEARCHING":
            base_stop()
            
        elif state == "APPROACH":
            base_stop()
            if lost_counter > LOST_DETECTION_TOLERANCE:
                print("[STATE] Lost target during approach. Resetting to SEARCH.")
                state = "SEARCHING"
                lost_counter = 0
                approach_start_area = 0
                target_position = None
                target_lock_counter = 0
        
        # [MCL] BUG FIX: Do not 'pass' on PICKING.
        # This state is handled above.
        # elif state == "PICKING":
        #    pass 
        elif state not in ["LOCALIZING", "PICKING", "COMPLETE"]:
            base_stop()

# ===============================================================
# CLEANUP
# ===============================================================
cv2.destroyAllWindows()
print("[INFO] Controller finished.")