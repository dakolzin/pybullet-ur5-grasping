# PyBullet UR5 Grasp Pipeline

## Overview

This repository presents a research-oriented grasping pipeline for the **UR5 robotic manipulator with a Robotiq 85 gripper** in the PyBullet simulation environment.
The project focuses on generating and evaluating grasp candidates, executing pick-and-place motions, and enabling reproducible experiments for grasp planning research.

The pipeline supports antipodal grasp generation, force-closure evaluation, motion planning, and synthetic dataset creation for grasp analysis.

---

## Features

* UR5 + Robotiq 85 simulation in PyBullet
* Antipodal grasp candidate generation
* Force-closure evaluation module
* Pick-and-place execution pipeline
* Synthetic point cloud dataset generation
* Grasp logging and analysis tools
* Modular utilities for planning, perception, and control

---

## Repository Structure

```
.
├── pick_pipeline.py        # Main grasp execution pipeline
├── capture_cloud_dataset.py# Synthetic dataset generation
├── force_closure_module.py # Force-closure evaluation
├── utils/                  # Core utilities
│   ├── camera_utils.py
│   ├── cloud_utils.py
│   ├── grasp_gen_utils.py
│   ├── motion_utils.py
│   ├── planning_utils.py
│   └── robot_ur5_robotiq85.py
├── meshes/                 # Robot and object meshes
├── urdf/                   # URDF models
├── tools/                  # Analysis scripts
├── graspit_env.yml         # Conda environment
└── README.md
```

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/dakolzin/pybullet-ur5-grasping.git
cd pybullet-ur5-grasping
```

### 2. Create environment

```bash
conda env create -f graspit_env.yml
conda activate graspit
```

---

## Quick Start

### Run grasp pipeline

```bash
python pick_pipeline.py
```

### Generate synthetic dataset

```bash
python capture_cloud_dataset.py
```

### Visualize point clouds

```bash
python view_cloud.py
```

---

## Pipeline Description

The grasp pipeline consists of the following stages:

1. **Scene generation**

   * Object spawning in simulation
   * Camera capture of depth and point clouds

2. **Point cloud processing**

   * Filtering and downsampling
   * Surface normal estimation

3. **Grasp generation**

   * Antipodal grasp sampling
   * Candidate filtering

4. **Grasp evaluation**

   * Force-closure verification
   * Collision checking

5. **Motion planning**

   * Inverse kinematics
   * Joint trajectory planning

6. **Execution**

   * Gripper closure
   * Object lifting and placement

---

## Reproducibility

All experiments can be reproduced using the provided environment file:

```bash
conda env create -f graspit_env.yml
conda activate graspit
```

No external datasets are required — synthetic scenes can be generated using:

```bash
python capture_cloud_dataset.py
```

---

## Demo

Example grasp execution in PyBullet simulation.

*(Add GIF or video link here)*

---

## Applications

* Research in robotic grasp planning
* Evaluation of force-closure metrics
* Synthetic dataset generation for grasp learning
* Benchmarking grasp strategies

---

## Future Work

* Integration with learning-based grasp planners
* Partial point cloud grasping experiments
* Real robot deployment
* Benchmark dataset publication

---

## License

This project is intended for research and educational use.
