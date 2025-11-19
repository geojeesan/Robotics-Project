import numpy as np

class ParticleFilter:
    """
    A simple Monte Carlo Localization (Particle Filter) class.
    
    Assumes map_data is a 2D numpy array where:
    - 0 = free space
    - 1 = occupied (wall)
    
    Assumes poses are [x, y, theta] where:
    - (x, y) are in world coordinates (meters)
    - theta is in radians
    """
    def __init__(self, num_particles, map_data, sensor_model_func, 
                 map_resolution=0.1, world_size_m=10.0):
        self.num_particles = num_particles
        self.map_data = map_data
        self.map_resolution = map_resolution  # meters per grid cell
        self.world_size_m = world_size_m
        self.map_origin_m = -world_size_m / 2.0
        
        # Pass in the complex sensor model function
        self.sensor_model = sensor_model_func
        
        # Initialize particles randomly within the world
        self.particles = self.init_particles()
        
        # Weights are uniform to start
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Motion model noise parameters (tune these)
        self.MOTION_NOISE_STD_DEV = {
            'x': 0.05,  # m noise per m moved
            'y': 0.05,  # m noise per m moved
            'theta': 0.01 # rad noise per rad turned
        }

    def init_particles(self):
        """Randomly distribute particles across the map in free space."""
        particles = np.empty((self.num_particles, 3))
        
        # Simple random initialization across the whole world
        # A better init would be to sample *only* from free space
        particles[:, 0] = np.random.uniform(self.map_origin_m, self.map_origin_m + self.world_size_m, self.num_particles)
        particles[:, 1] = np.random.uniform(self.map_origin_m, self.map_origin_m + self.world_size_m, self.num_particles)
        particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)
        
        return particles

    def predict(self, u, dt):
        """
        Move all particles based on the motion command u = [vx, vy, omega].
        This is the "Motion Model".
        """
        vx, vy, omega = u
        
        for i in range(self.num_particles):
            pose = self.particles[i]
            theta = pose[2]
            
            # Kinematic model for omni-wheel base
            dx = (vx * np.cos(theta) - vy * np.sin(theta)) * dt
            dy = (vx * np.sin(theta) + vy * np.cos(theta)) * dt
            dtheta = omega * dt
            
            # Add motion noise (proportional to movement)
            # This is a simplified noise model
            noise_x = np.random.normal(0.0, abs(dx) * self.MOTION_NOISE_STD_DEV['x'] + 0.005)
            noise_y = np.random.normal(0.0, abs(dy) * self.MOTION_NOISE_STD_DEV['y'] + 0.005)
            noise_theta = np.random.normal(0.0, abs(dtheta) * self.MOTION_NOISE_STD_DEV['theta'] + 0.002)

            # Update particle pose
            self.particles[i, 0] += dx + noise_x
            self.particles[i, 1] += dy + noise_y
            self.particles[i, 2] += dtheta + noise_theta
            
            # Normalize angle
            self.particles[i, 2] = np.mod(self.particles[i, 2] + np.pi, 2 * np.pi) - np.pi

    def update(self, z):
        """
        Update particle weights based on the sensor reading z (e.g., lidar scan).
        This is the "Sensor Update" step.
        """
        if z is None:
            return

        for i in range(self.num_particles):
            pose = self.particles[i]
            
            # Call the provided sensor_model function
            # This function calculates p(z | pose, map)
            likelihood = self.sensor_model(
                z, 
                pose, 
                self.map_data, 
                self.map_resolution,
                self.map_origin_m
            )
            
            self.weights[i] *= likelihood

        # Normalize weights
        self.weights += 1.e-300  # avoid division by zero
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights /= total_weight
        else:
            # All particles have zero weight - this is bad!
            # Reset weights to uniform.
            print("[WARN] Particle filter divergence! All weights are zero. Resetting.")
            self.weights = np.ones(self.num_particles) / self.num_particles

    def resample(self):
        """
        Resample new particles based on their weights (low-variance sampler).
        """
        # Check for particle deprivation (N_eff < N/2)
        n_eff = 1.0 / np.sum(np.square(self.weights))
        
        # Resample if effective particle count is too low
        if n_eff < self.num_particles / 2.0:
            indices = np.random.choice(
                np.arange(self.num_particles),
                size=self.num_particles,
                replace=True,
                p=self.weights
            )
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.num_particles)

    def get_estimated_pose(self):
        """
        Return the mean pose of the particle cloud.
        Handles angle averaging correctly.
        """
        avg_x = np.average(self.particles[:, 0], weights=self.weights)
        avg_y = np.average(self.particles[:, 1], weights=self.weights)
        
        # Average angles by converting to vectors
        avg_cos = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        avg_sin = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        avg_theta = np.arctan2(avg_sin, avg_cos)
        
        return np.array([avg_x, avg_y, avg_theta])

    def is_converged(self, threshold_std_m):
        """
        Check if the particle cloud has converged to a small area.
        Returns True if standard deviation of x and y is below threshold.
        """
        std_x = np.std(self.particles[:, 0])
        std_y = np.std(self.particles[:, 1])
        
        return std_x < threshold_std_m and std_y < threshold_std_m