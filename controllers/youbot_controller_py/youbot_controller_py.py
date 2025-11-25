from controller import Robot, Keyboard, Motor, Camera, Lidar, GPS, Display
import math
import sys
import numpy as np

# Constants
TIME_STEP = 32
BASE_SPEED = 4.0
MAX_SPEED = 0.3
SPEED_INCREMENT = 0.05
WHEEL_RADIUS = 0.05
LX = 0.228
LY = 0.158

# GPS Config
GPS_NOISE_STD = 0.5 

# Map Config
MAP_RESOLUTION = 0.1
MAP_ORIGIN_X = -15.0
MAP_ORIGIN_Y = -15.0
NUM_PARTICLES = 200

# Helper Classes

class ArmHeight:
    ARM_RESET = 2
    ARM_MAX_HEIGHT = 7
    PRESETS = {
        2: [1.57, -2.635, 1.78, 0.0],
        0: [0.92, 0.42, 1.78, 0.0],
        6: [-0.97, -1.55, -0.61, 0.0]
    }

class ArmOrientation:
    ARM_FRONT = 3
    ARM_MAX_SIDE = 7
    PRESETS = { 3: 0.0 }

class YoubotGripper:
    def __init__(self, robot):
        self.f1, self.f2 = robot.getDevice("finger1"), robot.getDevice("finger2")
        if self.f1: self.f1.setVelocity(0.03); self.f2.setVelocity(0.03)
    def grip(self):
        if self.f1: self.f1.setPosition(0.0); self.f2.setPosition(0.0)

class YoubotArm:
    def __init__(self, robot):
        self.el = [robot.getDevice(f"arm{i}") for i in range(1, 6)]
        self.current_height = ArmHeight.ARM_RESET
        self.current_orientation = ArmOrientation.ARM_FRONT
        if self.el[1]: self.el[1].setVelocity(0.5)
        self.reset()
    def reset(self):
        self.set_height(ArmHeight.ARM_RESET)
        self.set_orientation(ArmOrientation.ARM_FRONT)
    def set_height(self, h):
        if h in ArmHeight.PRESETS:
            vals = ArmHeight.PRESETS[h]
            for i in range(4):
                if self.el[i+1]: self.el[i+1].setPosition(vals[i])
        self.current_height = h
    def set_orientation(self, o):
        if o in ArmOrientation.PRESETS and self.el[0]:
            self.el[0].setPosition(ArmOrientation.PRESETS[o])
        self.current_orientation = o
    def increase_height(self): self.set_height(min(self.current_height + 1, ArmHeight.ARM_MAX_HEIGHT - 1))
    def decrease_height(self): self.set_height(max(self.current_height - 1, 0))
    def increase_orientation(self): self.set_orientation(min(self.current_orientation + 1, ArmOrientation.ARM_MAX_SIDE - 1))
    def decrease_orientation(self): self.set_orientation(max(self.current_orientation - 1, 0))

class YoubotBase:
    def __init__(self, robot):
        self.w = [robot.getDevice(f"wheel{i}") for i in range(1, 5)]
        for w in self.w: w.setPosition(float('inf')); w.setVelocity(0.0)
        self.vx, self.vy, self.omega = 0.0, 0.0, 0.0
    def update(self):
        g = LX + LY
        s = [(1/WHEEL_RADIUS)*(self.vx + self.vy + g*self.omega),
             (1/WHEEL_RADIUS)*(self.vx - self.vy - g*self.omega),
             (1/WHEEL_RADIUS)*(self.vx - self.vy + g*self.omega),
             (1/WHEEL_RADIUS)*(self.vx + self.vy - g*self.omega)]
        for i in range(4): self.w[i].setVelocity(s[i])
    def reset(self): self.vx=0; self.vy=0; self.omega=0; self.update()

# Visualizer
class ParticleVisualizer:
    def __init__(self, robot, map_grid, display_name="particle_display"):
        self.display = robot.getDevice(display_name)
        self.map_grid = map_grid
        if self.display:
            self.width = self.display.getWidth()
            self.height = self.display.getHeight()
            self.bg_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            h, w = map_grid.shape
            limit_h, limit_w = min(h, self.height), min(w, self.width)
            
            for y in range(limit_h):
                for x in range(limit_w):
                    val = 255 if map_grid[y, x] > 0 else 0
                    self.bg_image[self.height - 1 - y, x] = [val, val, val]
            
            self.ir = self.display.imageNew(self.bg_image.tobytes(), Display.RGB, self.width, self.height)
            self.display.imagePaste(self.ir, 0, 0, False)

    def world_to_screen(self, wx, wy):
        px = int((wx - MAP_ORIGIN_X) / MAP_RESOLUTION)
        py = int((wy - MAP_ORIGIN_Y) / MAP_RESOLUTION)
        px = max(0, min(px, self.width - 1))
        py = max(0, min(py, self.height - 1))
        return px, self.height - 1 - py

    def update(self, particles, est_x, est_y, est_th, gps_x=None, gps_y=None):
        if not self.display: return
        self.display.imagePaste(self.ir, 0, 0, False)
        
        # Particles (Red)
        self.display.setColor(0xFF0000)
        for p in particles:
            px, py = self.world_to_screen(p[0], p[1])
            self.display.drawPixel(px, py)

        # GPS (Blue Cross)
        if gps_x is not None:
            self.display.setColor(0x0000FF)
            gx, gy = self.world_to_screen(gps_x, gps_y)
            self.display.drawLine(gx-3, gy, gx+3, gy)
            self.display.drawLine(gx, gy-3, gx, gy+3)

        # Estimate (Green Cross)
        self.display.setColor(0x00FF00)
        ex, ey = self.world_to_screen(est_x, est_y)
        self.display.drawLine(ex-4, ey, ex+4, ey)
        self.display.drawLine(ex, ey-4, ex, ey+4)
        # Visualizer uses the raw theta (Compass) to draw the line direction
        hx, hy = self.world_to_screen(est_x + 0.6*math.cos(est_th), est_y + 0.6*math.sin(est_th))
        self.display.drawLine(ex, ey, hx, hy)

# Particle Filter
class ParticleFilter:
    def __init__(self, map_file, num_particles=200):
        self.num_particles = num_particles
        try:
            raw_map = np.load(map_file)
            self.map_grid = np.fliplr(raw_map)
            print(f"Map Loaded: {self.map_grid.shape}")
        except:
            self.map_grid = np.zeros((300, 300)); self.map_grid[0,:]=1; self.map_grid[-1,:]=1
        
        self.height, self.width = self.map_grid.shape
        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialize_randomly()
        
        # North (Map) corresponds to East (Compass)
        self.COMPASS_OFFSET = math.pi / 2

    def initialize_randomly(self):
        min_x = MAP_ORIGIN_X; max_x = MAP_ORIGIN_X + (self.width * MAP_RESOLUTION)
        min_y = MAP_ORIGIN_Y; max_y = MAP_ORIGIN_Y + (self.height * MAP_RESOLUTION)
        self.particles[:, 0] = np.random.uniform(min_x, max_x, self.num_particles)
        self.particles[:, 1] = np.random.uniform(min_y, max_y, self.num_particles)
        self.particles[:, 2] = np.random.uniform(-math.pi, math.pi, self.num_particles)

    def motion_update(self, vx, vy, omega, dt):
        # noise
        sigma_xy = 0.02 
        sigma_th = 0.03 

        theta_map = self.particles[:, 2] + self.COMPASS_OFFSET
        
        c, s = np.cos(theta_map), np.sin(theta_map)
        dx = (vx * c - vy * s) * dt
        dy = (vx * s + vy * c) * dt
        
        self.particles[:, 0] += dx + np.random.normal(0, sigma_xy, self.num_particles)
        self.particles[:, 1] += dy + np.random.normal(0, sigma_xy, self.num_particles)
        self.particles[:, 2] += omega * dt + np.random.normal(0, sigma_th, self.num_particles)

    def sensor_update_lidar(self, lidar_ranges, lidar_fov):
        skip = 20
        ranges = np.array(lidar_ranges[::skip])
        valid = np.isfinite(ranges) & (ranges < 5.0)
        ranges = ranges[valid]
        if len(ranges) == 0: return

        angles = np.linspace(-lidar_fov/2, lidar_fov/2, len(lidar_ranges))[::skip][valid]
        angles = angles + self.COMPASS_OFFSET
        
        for i in range(self.num_particles):
            px, py, pth = self.particles[i]
            ga = pth + angles
            ex = px + ranges * np.cos(ga)
            ey = py + ranges * np.sin(ga)
            
            mx = ((ex - MAP_ORIGIN_X) / MAP_RESOLUTION).astype(int)
            my = ((ey - MAP_ORIGIN_Y) / MAP_RESOLUTION).astype(int)
            
            # Bounds check
            in_b = (mx >= 0) & (mx < self.width) & (my >= 0) & (my < self.height)
            
            # Map Logic: 1=Wall, 0=Free
            # If ray lands on 1, add score.
            score = np.sum(self.map_grid[my[in_b], mx[in_b]])
            
            # Add small baseline to avoid weight=0
            self.weights[i] *= (score + 1e-5)

        self.normalize_weights()

    def sensor_update_gps(self, gps_x, gps_y):
        dx = self.particles[:, 0] - gps_x
        dy = self.particles[:, 1] - gps_y
        dist_sq = dx**2 + dy**2
        likelihood = np.exp(-dist_sq / (2 * GPS_NOISE_STD**2))
        self.weights *= (likelihood + 1e-10)
        self.normalize_weights()

    def normalize_weights(self):
        s = np.sum(self.weights)
        if s == 0 or np.isnan(s): self.weights.fill(1.0/self.num_particles)
        else: self.weights /= s

    def resample(self):
        # 1. Standard Resampling
        ind = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[ind]
        self.weights.fill(1.0/self.num_particles)

        # 2. Replace 5% of particles with random ones to handle "Kidnapped Robot"
        num_inject = int(self.num_particles * 0.05) # 5%
        if num_inject > 0:
            min_x = MAP_ORIGIN_X; max_x = MAP_ORIGIN_X + (self.width * MAP_RESOLUTION)
            min_y = MAP_ORIGIN_Y; max_y = MAP_ORIGIN_Y + (self.height * MAP_RESOLUTION)
            
            random_indices = np.random.choice(self.num_particles, num_inject, replace=False)
            
            self.particles[random_indices, 0] = np.random.uniform(min_x, max_x, num_inject)
            self.particles[random_indices, 1] = np.random.uniform(min_y, max_y, num_inject)
            self.particles[random_indices, 2] = np.random.uniform(-math.pi, math.pi, num_inject)

    def get_estimate(self):
        mx = np.mean(self.particles[:, 0])
        my = np.mean(self.particles[:, 1])
        mth = np.arctan2(np.mean(np.sin(self.particles[:, 2])), np.mean(np.cos(self.particles[:, 2])))
        return mx, my, mth

# --- Main Controller ---
class YoubotController:
    def __init__(self):
        self.robot = Robot()
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(TIME_STEP)

        self.base = YoubotBase(self.robot)
        self.arm = YoubotArm(self.robot)
        self.gripper = YoubotGripper(self.robot)
        
        self.lidar = self.robot.getDevice("lidar")
        if self.lidar: self.lidar.enable(TIME_STEP); self.lidar.enablePointCloud()
        
        self.gps = self.robot.getDevice("gps")
        if self.gps: self.gps.enable(TIME_STEP)

        self.pf = ParticleFilter("final_map.npy", NUM_PARTICLES)
        self.vis = ParticleVisualizer(self.robot, self.pf.map_grid)

    def run(self):
        self.base.reset(); self.gripper.grip(); self.arm.reset()
        print("Controller Started (Inverted Map + Orientation Fix).")
        
        pc = 0
        while self.robot.step(TIME_STEP) != -1:
            c = self.keyboard.getKey()
            if c >= 0 and c != pc:
                is_shift = (c & Keyboard.SHIFT); key = (c & ~Keyboard.SHIFT)
                if is_shift:
                    if key==Keyboard.UP: self.arm.increase_height()
                    elif key==Keyboard.DOWN: self.arm.decrease_height()
                else:
                    if key==Keyboard.UP: self.base.vx += SPEED_INCREMENT
                    elif key==Keyboard.DOWN: self.base.vx -= SPEED_INCREMENT
                    elif key==Keyboard.LEFT: self.base.vy += SPEED_INCREMENT
                    elif key==Keyboard.RIGHT: self.base.vy -= SPEED_INCREMENT
                    elif key==ord('n'): self.base.omega += SPEED_INCREMENT
                    elif key==ord('m'): self.base.omega -= SPEED_INCREMENT
                    elif key==ord(' '): self.base.reset(); self.arm.reset()
                
                self.base.vx = max(min(self.base.vx, MAX_SPEED), -MAX_SPEED)
                self.base.vy = max(min(self.base.vy, MAX_SPEED), -MAX_SPEED)
                self.base.omega = max(min(self.base.omega, MAX_SPEED), -MAX_SPEED)
                self.base.update()
            pc = c

            dt = TIME_STEP/1000.0
            if abs(self.base.vx)>0.01 or abs(self.base.vy)>0.01 or abs(self.base.omega)>0.01:
                self.pf.motion_update(self.base.vx, self.base.vy, self.base.omega, dt)

            #GPS Fusion
            gx, gy = None, None
            if self.gps:
                vals = self.gps.getValues()
                gx, gy = vals[0], vals[1]
                self.pf.sensor_update_gps(gx, gy)

            # Lidar (If arm up)
            # if self.lidar and self.arm.current_height == ArmHeight.ARM_RESET:
            r = self.lidar.getRangeImage()
            if r: self.pf.sensor_update_lidar(r, self.lidar.getFov())
            
            self.pf.resample()
            
            ex, ey, eth = self.pf.get_estimate()
            self.vis.update(self.pf.particles, ex, ey, eth, gx, gy)
            # self.vis.update(self.pf.particles, ex, ey, eth)

if __name__ == "__main__":
    YoubotController().run()