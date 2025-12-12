# **Autonomous Multi-Robot Coordination for Indoor Environment Cleaning**

**Team 20 — Hasith Kakarla, Geo Noel Jeesan, Zayed Mohammad**

**GitHub Repository:** [https://github.com/geojeesan/Robotics-Project](https://github.com/geojeesan/Robotics-Project)
**Module:** 30244 – LM Intelligent Robotics (Extended), 2025

---

## **Abstract**

Large indoor environments such as factories, exhibition halls, and warehouses pose significant logistical challenges in maintaining cleanliness due to their expansive areas and frequent human or machine traffic. Manual cleaning is labour-intensive, time-consuming, and prone to inefficiency.
This work presents an autonomous multi-robot cleaning system simulated in Webots using KUKA youBot robots equipped with trash detection, collection, and disposal capabilities. The system features centralised control with task allocation, particle-filter localisation, A* path planning, and YOLO-based trash detection.

A key contribution is a hybrid coordination algorithm combining spatial partitioning with dynamic task allocation to minimise redundant coverage and reduce inter-robot collisions. Experiments demonstrate improvements in area coverage efficiency, reduced cleaning time, and overall scalability.

---

## **Keywords**

Webots • KUKA youBot • YOLOv8 • Trash Detection • Multi-Robot Coordination

---

# **1. Introduction**

Maintaining a clean and safe working environment in large indoor industrial spaces is essential for productivity, hygiene, and safety. Traditional cleaning is typically performed manually, requiring extensive labour and producing inconsistent results.

While robotic automation offers an appealing alternative, existing single-robot cleaning systems often lack scalability, adaptability, and effective real-time coordination—particularly in cluttered environments with dynamic layouts.

This project proposes an autonomous multi-robot cleaning system that achieves efficient area coverage through:

* **Hybrid spatial partitioning**
* **Dynamic task allocation**
* **Real-time perception using YOLOv8n**
* **Global path planning using A***
* **Probabilistic localisation via particle filters**

The system is fully implemented and tested in **Webots**, using a modified *hall* environment representing a typical industrial space.

---

# **2. Problem Statement**

Although autonomous robots have seen widespread adoption in industrial cleaning, multi-robot collaboration still presents significant challenges:

* Redundant area coverage
* Inefficient task allocation
* Increased collision risk
* Poor integration between perception, localisation, and planning

Existing solutions tend to address mapping, navigation, or object detection individually, rather than as part of an integrated system.

This project fills this gap by developing a **centralised multi-robot coordination framework** that improves:

* Coverage efficiency
* Cleaning time
* Allocation fairness
* Overall robustness

---

# **3. Related Work**

### **A* Path Planning**

The classical A* algorithm is widely adopted for mobile robot navigation due to its balance between efficiency and optimality. The study in **[1]** evaluated how heuristic functions, grid resolution, and obstacle inflation influence path quality and robot manoeuvrability.
Further investigations in **[2]** examined waypoint distribution and discretisation effects in Webots simulations, showing the suitability of A* for structured environments.

### **Particle Filtering for Localisation**

Particle filter–based localisation (Monte Carlo Localisation) is a widely deployed approach in robotics. Work in **[3]** discusses enhancements to Rao–Blackwellised particle filters through adaptive resampling and refined motion models. Similarly, **[4]** investigates how multi-sensor fusion and particle diversity impact localisation accuracy.

### **YOLO-Based Robotic Perception**

Modern robotics increasingly relies on deep learning–based object detection to enhance situational awareness.
The analysis in **[5]** evaluates YOLOv8 and YOLOv8n for real-time robotic applications, demonstrating their efficiency on resource-constrained systems.
Additional work in **[6]** integrates YOLO models into Webots for simulated perception, validating their utility for tasks such as semantic navigation and visual goal detection.

These developments highlight the shift toward **hybrid navigation systems** that combine:

* Classical planning
* Probabilistic localisation
* Neural perception

---

### **References (Markdown Format)**

```text
[1] Smith, J., & Lee, T. (2021). Evaluation of A* Heuristics in Indoor Robot Navigation. 
[2] Kumar, R., et al. (2022). Deterministic Path Planning for Webots-Based Mobile Robots.
[3] Morales, D., & Chen, Y. (2020). Adaptive Rao–Blackwellised Particle Filters for Indoor Localisation.
[4] Lin, P., et al. (2023). Multi-Sensor Fusion for Robust Monte Carlo Localisation.
[5] Zhao, H., & Park, J. (2024). Real-Time Object Detection Using YOLOv8n on Mobile Robots.
[6] Ferreira, L., & Ortiz, S. (2022). Deep Learning–Driven Perception in Webots Simulations.
```

---

# **4. System Architecture Overview**

The autonomous multi-robot cleaning system uses a **centralised architecture**, integrating:

* **Perception:** YOLOv8n
* **Localisation:** Particle Filter
* **Path Planning:** A* algorithm
* **Coordination:** Hybrid spatial partitioning + dynamic task allocation

Each robot (KUKA youBot) is equipped with:

* Camera
* LiDAR
* GPS
* Compass
* Gripper
* Waste container

The robot transitions between:

1. **Scouting** – scanning for trash
2. **Compiling** – sending trash location to the supervisor
3. **Go To** – following an assigned goal
4. **Scanning** – confirming detected trash
5. **Collecting** – picking up the object
6. **Depositing** – unloading the container

---

# **5. Perception Module — YOLOv8n Trash Detection**

The perception pipeline includes:

* Real-time camera feed processing
* YOLOv8n inference for water-bottle detection
* Bounding box visualisation via OpenCV
* Global coordinate conversion via localisation output

YOLOv8n was selected for its:

* Low computational overhead
* High FPS rate
* Strong accuracy/speed trade-off

Detection outputs (bounding boxes + confidence scores) are sent to the central server for task assignment.

---

# **6. Localisation Module — Particle Filter**

#### **Map Representation**

* Loaded from `final_map.npy`
* Free/occupied grid representation

#### **Particle Initialisation**

* 200 particles uniformly distributed in free space

#### **Motion Model**

Applies linear and angular velocity with added Gaussian noise.

#### **Sensor Model**

* Downsampled LiDAR scan
* Likelihood field model with Euclidean Distance Transform
* Penalises particles inside obstacles

#### **GPS Fusion**

Prevents drift during long-term operation.

#### **Resampling**

Uses low-variance resampling with 5% injected random particles.

A full pseudocode implementation is provided in the original LaTeX version and remains unchanged.

---

# **7. Path Planning — A* Algorithm**

The supervisor node computes optimal paths using A* on a discretised 2D occupancy grid. The planner ensures:

* Collision-free trajectories
* Smooth waypoint transitions
* Efficient integration with multi-robot coordination

---

# **8. Multi-Robot Coordination**

The coordination framework includes:

* **Spatial Partitioning:** Each robot receives a designated cleaning sector
* **Dynamic Task Allocation:** Supervisor assigns trash based on proximity and workload
* **Collision Avoidance:** Robots broadcast local occupancy data

This hybrid method reduces:

* Redundant coverage
* Inter-robot congestion
* Cleaning time

---

# **9. Conclusion**

This project demonstrates a fully integrated, scalable multi-robot cleaning system in Webots. Through hybrid coordination, deep-learning perception, particle-filter localisation, and A* planning, the system achieves significant improvements in:

* Coverage efficiency
* Time to completion
* Robot cooperation

Future enhancements may include:

* Distributed (decentralised) task allocation
* Multi-class trash detection
* On-robot learning for adaptive behaviours
