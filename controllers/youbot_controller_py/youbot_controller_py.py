from controller import Robot
import cv2
import numpy as np

# ===============================================================
# PARAMETERS
# ===============================================================
TIME_STEP = 64
DEBUG_VIEW = True
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
IDEAL_STOP_Y_RATIO = 0.80
FINAL_STOP_Y_RATIO = 0.90

# Robot geometry for omnidirectional base
WHEEL_RADIUS = 0.05
LX = 0.228
LY = 0.158

# ===============================================================
# INITIALIZATION
# ===============================================================
robot = Robot()
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

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
    """Reset arm to default position"""
    arm_joints[0].setPosition(0.0)
    arm_joints[1].setPosition(1.57)
    arm_joints[2].setPosition(-2.635)
    arm_joints[3].setPosition(1.78)
    arm_joints[4].setPosition(0.0)

def arm_pick_position():
    """Set arm to picking position"""
    arm_joints[0].setPosition(0.0)
    arm_joints[1].setPosition(-0.97)
    arm_joints[2].setPosition(-1.55)
    arm_joints[3].setPosition(-0.61)
    arm_joints[4].setPosition(0.0)

def arm_lift_position():
    """Set arm to lift position"""
    arm_joints[0].setPosition(2.949)
    arm_joints[1].setPosition(0.92)
    arm_joints[2].setPosition(0.42)
    arm_joints[3].setPosition(1.78)
    arm_joints[4].setPosition(0.0)

def gripper_open():
    """Open the gripper"""
    if has_gripper:
        gripper.setPosition(0.025)

def gripper_close():
    """Close the gripper to grip object"""
    if has_gripper:
        gripper.setPosition(0.0)

# ===============================================================
# DETECT RED OBJECT
# ===============================================================
def detect_red_object(image, target_position=None, target_area=None):
    """Detect red object in camera frame using OpenCV and return the best match."""
    img = np.frombuffer(image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color range
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find all valid objects
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
            # Tracking a specific object - find the one closest to our target
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

                # Relax area check when close
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
            # No target locked yet - pick the one closest to center
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

        # Visualization
        if DEBUG_VIEW:
            for obj in valid_objects:
                cx, cy, area, x, y, w, h = obj
                if tracked_obj and obj == tracked_obj:
                    color = (0, 255, 0)
                    thickness = 2
                else:
                    color = (0, 255, 255)
                    thickness = 1

                cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
                cv2.circle(img, (cx, cy), 3, color, -1)
                if tracked_obj and obj == tracked_obj:
                    cv2.putText(img, f"TRACKED Area:{int(area)}", (x, y - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.line(img, (img_center_x, 0), (img_center_x, camera.getHeight()), (255, 0, 0), 1)

            # Draw the ideal stopping Y line
            ideal_y_stop = int(img_height * IDEAL_STOP_Y_RATIO)
            final_y_stop = int(img_height * FINAL_STOP_Y_RATIO)
            cv2.line(img, (0, ideal_y_stop), (img_width, ideal_y_stop), (255, 0, 0), 1)
            cv2.line(img, (0, final_y_stop), (img_width, final_y_stop), (0, 0, 255), 1)

            if target_position:
                tx, ty = target_position
                cv2.circle(img, (int(tx), int(ty)), 8, (255, 0, 255), 2)

            if detected:
                error = detected[0] - img_center_x
                cv2.putText(img, f"Error:{int(error)}px", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img, f"Objects:{len(valid_objects)}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if DEBUG_VIEW:
        cv2.imshow("Camera Feed", img)
        cv2.waitKey(1)
    return detected

# ===============================================================
# MAIN LOOP
# ===============================================================

# Initialize arm and gripper
arm_reset()
gripper_open()

frame_count = 0
lost_counter = 0
state = "SEARCHING"
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

while robot.step(TIME_STEP) != -1:
    frame_count += 1
    if frame_count <= CAMERA_WARMUP_STEPS:
        continue
    if frame_count % FRAME_SKIP != 0:
        continue

    image = camera.getImage()
    if not image:
        continue

    is_locked = target_lock_counter > TARGET_LOCK_THRESHOLD
    detection = detect_red_object(image,
                                 target_position if is_locked else None,
                                 approach_start_area if is_locked and approach_start_area > 0 else None)

    if detection:
        center_x, center_y, area = detection
        width = camera.getWidth()
        height = camera.getHeight()
        error = center_x - width // 2
        lost_counter = 0

        # Tracking and Locking Logic
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

        if state == "SEARCHING":
            if target_lock_counter > TARGET_LOCK_THRESHOLD:
                state = "APPROACH"
                approach_start_area = area
                last_known_position = (center_x, center_y)
                last_known_error = error
                last_known_area = area
                forward_direction = 1
                area_trend_counter = 0
                last_area_for_motion = 0
                close_confirmation_counter = 0
            else:
                if abs(error) > 20:
                    turn_speed = np.clip(-error / 40.0, -0.4, 0.4)
                    base_turn(turn_speed)
                else:
                    base_stop()

        elif state == "APPROACH":
            # Update last known values
            last_known_position = (center_x, center_y)
            last_known_error = error
            last_known_area = area
            lost_counter = 0

            required_area = TARGET_AREA
            if approach_start_area > 0:
                required_area = max(TARGET_AREA, approach_start_area * APPROACH_AREA_MULTIPLIER)

            ideal_y_stop = height * IDEAL_STOP_Y_RATIO
            final_y_stop = height * FINAL_STOP_Y_RATIO

            # CRITICAL EMERGENCY STOP CHECK
            if center_y > final_y_stop:
                base_stop()
                print(f"[STATE] 🛑 EMERGENCY STOP: Object center_y ({center_y:.0f}) passed final y stop ({final_y_stop:.0f}). Starting pickup sequence.")
                state = "PICKING"
                pickup_timer = 0
                lost_counter = 0
                continue

            # Proportional gain for speed
            if center_y < ideal_y_stop and required_area > approach_start_area:
                speed_ratio = (required_area - area) / (required_area - approach_start_area)
                speed_ratio = np.clip(speed_ratio, 0.0, 1.0)
                forward_speed = (0.15 - STOPPING_SPEED) * speed_ratio + STOPPING_SPEED
            else:
                forward_speed = STOPPING_SPEED * 0.5

            forward_speed *= forward_direction

            # Check for the stopping condition: object reached the ideal Y position
            is_at_ideal_y = center_y >= ideal_y_stop

            # FIRST: Center the object
            if abs(error) > 15:
                turn_speed = np.clip(-error / 30.0, -0.5, 0.5)
                base_turn(turn_speed)
                close_confirmation_counter = 0
            elif not is_at_ideal_y:
                # Object is centered and we're not close enough yet - move forward

                # Area trend check (ensure we're not moving away)
                if last_area_for_motion == 0:
                    last_area_for_motion = area
                if area < last_area_for_motion * 0.92:
                    area_trend_counter -= 1
                elif area > last_area_for_motion * 1.05:
                    area_trend_counter = min(area_trend_counter + 1, 5)
                else:
                    if area_trend_counter < 0:
                        area_trend_counter += 1
                    elif area_trend_counter > 0:
                        area_trend_counter -= 1

                if area_trend_counter <= -3:
                    forward_direction *= -1
                    area_trend_counter = 0
                    print(f"[ADAPT] Area shrinking while advancing. Reversing forward direction to {forward_direction:+d}.")

                # Move forward with calculated proportional speed
                base_forward(forward_speed)
                last_area_for_motion = area
                close_confirmation_counter = 0
                if frame_count % 50 == 0:
                    print(f"[APPROACH] Moving toward target... speed: {forward_speed:.2f}m/s, center_y: {center_y:.0f} (ideal: {ideal_y_stop:.0f}), area: {area:.0f}")
            else:
                # Close enough based on Y-position! Require confirmation
                close_confirmation_counter += 1
                if close_confirmation_counter >= APPROACH_CONFIRMATION_FRAMES:
                    base_stop()
                    print(f"[STATE] Trash reached! Center Y: {center_y:.0f}. Starting pickup sequence...")
                    state = "PICKING"
                    pickup_timer = 0
                    lost_counter = 0
                else:
                    # Slow crawl forward while confirming
                    base_forward(STOPPING_SPEED * 0.5 * forward_direction)
                    if frame_count % 20 == 0:
                        print(f"[APPROACH] Confirming proximity ({close_confirmation_counter}/{APPROACH_CONFIRMATION_FRAMES}) - center_y: {center_y:.0f}")

        elif state == "PICKING":
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
                base_stop()

        elif state == "COMPLETE":
            pickup_timer += 1
            if pickup_timer == 1:
                arm_reset()
                gripper_open()
                target_position = None
                target_lock_counter = 0
                last_known_position = None
                last_known_error = 0
                last_known_area = 0
                approach_start_area = 0
                lost_counter = 0
            elif pickup_timer > 30:
                state = "SEARCHING"
                pickup_timer = 0
            else:
                base_stop()
    else:
        # Lost Detection Logic
        lost_counter += 1

        if state == "SEARCHING":
            base_turn(0.3)
        elif state == "APPROACH":
            is_locked = target_lock_counter > TARGET_LOCK_THRESHOLD
            if is_locked and last_known_position is not None:
                if lost_counter <= LOST_DETECTION_TOLERANCE:
                    if abs(last_known_error) > 15:
                        turn_speed = np.clip(-last_known_error / 35.0, -0.4, 0.4)
                        base_turn(turn_speed)
                    elif last_known_area < TARGET_AREA:
                        base_forward(0.12 * forward_direction)
                    else:
                        base_stop()
                        if lost_counter > 30 and last_known_area >= TARGET_AREA * 0.8:
                            state = "PICKING"
                            pickup_timer = 0
                            lost_counter = 0
                else:
                    base_stop()
                    state = "SEARCHING"
                    lost_counter = 0
                    approach_start_area = 0
                    target_position = None
                    target_lock_counter = 0
            else:
                if lost_counter > 15:
                    base_stop()
                    state = "SEARCHING"
                    lost_counter = 0
                    approach_start_area = 0
                    target_position = None
                    target_lock_counter = 0
                else:
                    base_stop()
        elif state == "PICKING":
            pass
        else:
            base_stop()

# ===============================================================
# CLEANUP
# ===============================================================
cv2.destroyAllWindows()
print("[INFO] Controller finished.")