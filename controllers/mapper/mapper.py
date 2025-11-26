import numpy as np
from controller import Robot, Keyboard

TIME_STEP = 64

# --- Map Parameters ---
WORLD_SIZE_M = 30.0
MAP_RESOLUTION_M = 0.1
MAP_SIZE_PIXELS = int(WORLD_SIZE_M / MAP_RESOLUTION_M)
MAP_ORIGIN_M = -WORLD_SIZE_M / 2.0
MAP_FILE = 'map.npy'

# --- Log-Odds Parameters ---
LOG_ODDS_HIT = 0.5
LOG_ODDS_FREE = -0.7
LOG_ODDS_CLAMP_MAX = 5.0
LOG_ODDS_CLAMP_MIN = -5.0
THRESHOLD_OCC = 0.9
THRESHOLD_FREE = -0.5

# --- Create/Load Map ---
try:
    map_grid_log_odds = np.load(MAP_FILE + '.logodds.npy')
    if map_grid_log_odds.shape[0] != MAP_SIZE_PIXELS:
        print(f"Map size changed! Creating new {MAP_SIZE_PIXELS}x{MAP_SIZE_PIXELS} map.")
        map_grid_log_odds = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS), dtype=np.float32)
    else:
        print(f"Loaded existing log-odds map '{MAP_FILE}.logodds.npy'")
except FileNotFoundError:
    map_grid_log_odds = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS), dtype=np.float32)
    print(f"Created new log-odds map grid {MAP_SIZE_PIXELS}x{MAP_SIZE_PIXELS}")

# --- Robot Parameters ---
WHEEL_RADIUS = 0.05
LX = 0.228
LY = 0.158
MAX_SPEED = 0.7

# --- Helper Functions ---

def base_move(vx, vy, omega):
    speeds = [
        1 / WHEEL_RADIUS * (vx + vy + (LX + LY) * omega),
        1 / WHEEL_RADIUS * (vx - vy - (LX + LY) * omega),
        1 / WHEEL_RADIUS * (vx - vy + (LX + LY) * omega),
        1 / WHEEL_RADIUS * (vx + vy - (LX + LY) * omega)
    ]
    for i, wheel in enumerate(wheels):
        wheel.setVelocity(speeds[i])

def world_to_map(x_m, y_m):
    # [FIX] Using Y as the second map coordinate based on user finding
    px = int(np.floor((float(x_m) - MAP_ORIGIN_M) / MAP_RESOLUTION_M))
    py = int(np.floor((float(y_m) - MAP_ORIGIN_M) / MAP_RESOLUTION_M))
    return px, py

def bresenham_line_vectorized(x0, y0, x1, y1):
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    swapped = False
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        swapped = True

    dx = x1 - x0
    dy = abs(y1 - y0)
    error = dx / 2.0
    ystep = 1 if y0 < y1 else -1
    
    y = y0
    points = []
    for x in range(x0, x1 + 1):
        coord = (y, x) if steep else (x, y)
        points.append(coord)
        error -= dy
        if error < 0:
            y += ystep
            error += dx

    if swapped:
        points.reverse()
        
    points = np.array(points)
    if len(points) == 0:
        return np.array([]), np.array([])
        
    return points[:, 0], points[:, 1]

def update_map_vectorized(pose, lidar_scan, lidar_angles, num_rays, max_range):
    pose_x, pose_y, pose_theta = pose
    robot_px, robot_py = world_to_map(pose_x, pose_y)
    
    print(f"[DEBUG] Robot Grid: ({robot_px}, {robot_py})")

    scan = np.array(lidar_scan)[0:num_rays]
    
    valid_mask = (scan > 0.25) & (scan < max_range - 0.05) & (~np.isinf(scan))
    valid_scan = scan[valid_mask]
    valid_angles = lidar_angles[valid_mask]
    
    if len(valid_scan) == 0:
        return

    ray_angles_world = pose_theta + valid_angles
    
    # [FIX] Calculate hits in X-Y plane (using Y as vertical map axis)
    hit_x = pose_x + valid_scan * np.sin(ray_angles_world)
    hit_y = pose_y + valid_scan * np.cos(ray_angles_world)
    
    hit_px = np.floor((hit_x - MAP_ORIGIN_M) / MAP_RESOLUTION_M).astype(int)
    hit_py = np.floor((hit_y - MAP_ORIGIN_M) / MAP_RESOLUTION_M).astype(int)
    
    for i in range(len(hit_px)):
        hpx, hpy = hit_px[i], hit_py[i]
        
        line_x, line_y = bresenham_line_vectorized(robot_px, robot_py, hpx, hpy)
        
        if len(line_x) == 0: continue
            
        mask = (line_x >= 0) & (line_x < MAP_SIZE_PIXELS) & \
               (line_y >= 0) & (line_y < MAP_SIZE_PIXELS)
        
        line_x = line_x[mask]
        line_y = line_y[mask]
        
        if len(line_x) == 0: continue

        map_grid_log_odds[line_y[:-1], line_x[:-1]] += LOG_ODDS_FREE
        map_grid_log_odds[line_y[-1], line_x[-1]] += LOG_ODDS_HIT
        
    np.clip(map_grid_log_odds, LOG_ODDS_CLAMP_MIN, LOG_ODDS_CLAMP_MAX, out=map_grid_log_odds)


def save_final_map():
    print("\nConverting log-odds map to final 0,1,2 format...")
    final_map = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS), dtype=np.int8)
    final_map[map_grid_log_odds > THRESHOLD_OCC] = 1
    final_map[map_grid_log_odds < THRESHOLD_FREE] = 2
    
    np.save(MAP_FILE, final_map)
    np.save(MAP_FILE + '.logodds.npy', map_grid_log_odds)
    print(f"Final map saved to '{MAP_FILE}'.")


# --- Initialization ---
robot = Robot()
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
wheels = [robot.getDevice(name) for name in wheel_names]
for wheel in wheels:
    wheel.setPosition(float('inf'))
    wheel.setVelocity(0.0)

try:
    gps = robot.getDevice("gps")
    gps.enable(TIME_STEP)
    compass = robot.getDevice("compass")
    compass.enable(TIME_STEP)
    lidar = robot.getDevice("lidar")
    lidar.enable(TIME_STEP)
    lidar.enablePointCloud()
    
    LIDAR_MAX_RANGE = lidar.getMaxRange()
    LIDAR_NUM_RAYS = lidar.getHorizontalResolution()
    LIDAR_FOV = lidar.getFov()
    LIDAR_ANGLES = np.linspace(-LIDAR_FOV / 2.0, LIDAR_FOV / 2.0, LIDAR_NUM_RAYS)
except Exception as e:
    print(f"FATAL: Missing sensors. {e}")
    exit()

print("Mapper started.")
print("  Drive: W/A/S/D/Q/E")
print("  Scan (Compass): SPACE")
print("  Scan (Manual): I (North), J (West/Left), L (East/Right), M (South/Down)")
print("  Save: K")
last_key_pressed = -1
last_robot_px = -1
last_robot_py = -1

# --- Main Loop ---
try:
    while robot.step(TIME_STEP) != -1:
        key = keyboard.getKey()
        vx, vy, omega = 0.0, 0.0, 0.0
        
        if key == ord('W'): vx = MAX_SPEED
        elif key == ord('S'): vx = -MAX_SPEED
        elif key == ord('A'): vy = MAX_SPEED
        elif key == ord('D'): vy = -MAX_SPEED
        elif key == ord('Q'): omega = MAX_SPEED
        elif key == ord('E'): omega = -MAX_SPEED
        
        base_move(vx, vy, omega)
        
        scan_compass = (key == ord(' ') and last_key_pressed != ord(' '))
        scan_west   = (key == ord('I') and last_key_pressed != ord('I'))
        scan_north    = (key == ord('J') and last_key_pressed != ord('J'))
        scan_south    = (key == ord('L') and last_key_pressed != ord('L'))
        scan_east   = (key == ord('M') and last_key_pressed != ord('M'))
        
        save_pressed = (key == ord('K') and last_key_pressed != ord('K'))

        scan_angle = None
        scan_type = ""

        if scan_compass:
            scan_type = "Compass"
            compass_val = compass.getValues()
            
            cx = compass_val[0]
            cy = compass_val[1] # [FIX] Use Y from compass if GPS Y is used
            
            magnitude = np.sqrt(cx*cx + cy*cy)
            if magnitude > 0.001:
                cx /= magnitude
                cy /= magnitude
                scan_angle = np.arctan2(-cx, cy)
            else:
                scan_angle = 0.0

        elif scan_north:
            scan_type = "Fixed NORTH"
            scan_angle = 0.0
        elif scan_west:
            scan_type = "Fixed WEST"
            scan_angle = -np.pi / 2.0
        elif scan_east:
            scan_type = "Fixed EAST"
            scan_angle = np.pi / 2.0
        elif scan_south:
            scan_type = "Fixed SOUTH"
            scan_angle = np.pi

        if scan_angle is not None:
            base_move(0, 0, 0)
            
            gps_val = gps.getValues()
            pose_x = -gps_val[0]
            pose_y = gps_val[1] # [FIX] Use Y as the second map coordinate
            
            curr_px, curr_py = world_to_map(pose_x, pose_y)
            if curr_px == last_robot_px and curr_py == last_robot_py:
                print(f"[MAPPER] WARNING: Robot hasn't moved map pixels! (still {curr_px},{curr_py})")
                print(f"         Move further (at least >10cm) to see changes.")
            else:
                print(f"[MAPPER] Scanning ({scan_type})... Moved to grid ({curr_px}, {curr_py})")
                
            last_robot_px = curr_px
            last_robot_py = curr_py
            
            # Passing (x, y, theta) to vectorized map updater
            update_map_vectorized(
                (pose_x, pose_y, scan_angle),
                lidar.getRangeImage(),
                LIDAR_ANGLES,
                LIDAR_NUM_RAYS,
                LIDAR_MAX_RANGE
            )

        if save_pressed:
            save_final_map()

        last_key_pressed = key

finally:
    base_move(0, 0, 0)
    save_final_map()