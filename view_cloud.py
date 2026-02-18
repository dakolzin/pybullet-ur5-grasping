import numpy as np
import pybullet as p
import pybullet_data
import time
import random

PATH = "dataset_clouds/scene_000000.npz"  # поменяй номер

data = np.load(PATH)
pts = data["points_world"]

print("Points:", pts.shape)

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# Оси координат
p.addUserDebugLine([0,0,0], [0.2,0,0], [1,0,0], 3)  # X красный
p.addUserDebugLine([0,0,0], [0,0.2,0], [0,1,0], 3)  # Y зелёный
p.addUserDebugLine([0,0,0], [0,0,0.2], [0,0,1], 3)  # Z синий

# Рисуем НЕ ВСЕ точки
MAX_PTS = 1000
idx = np.random.choice(len(pts), min(MAX_PTS, len(pts)), replace=False)

for i in idx:
    x, y, z = pts[i]
    p.addUserDebugLine(
        [x, y, z],
        [x, y, z + 0.002],
        [0, 1, 0],
        1,
        lifeTime=0
    )

while True:
    p.stepSimulation()
    time.sleep(0.01)
