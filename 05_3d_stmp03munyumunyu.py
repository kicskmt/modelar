
# むにゅむにゅ表示
# 数を増やしてさらにムニュムニュ表示

import open3d as o3d
import numpy as np
import time
import random
import copy

# STL読み込み
base_mesh = o3d.io.read_triangle_mesh("zWitch_Hat.stl")
base_mesh.compute_vertex_normals()
base_mesh.paint_uniform_color([1.0, 0.84, 0.0])
vertices0 = np.asarray(base_mesh.vertices)
center = vertices0.mean(axis=0)
vertices0 -= center

# 複数オブジェクト
num_objects = 1
objects = []

for _ in range(num_objects):
    mesh = copy.deepcopy(base_mesh)
    vertices = vertices0.copy()
    pos = np.array([
        random.uniform(-3.0, 3.0),
        random.uniform(0.5, 3.0),
        random.uniform(-3.0, 3.0)
    ])
    rotation = np.eye(3)  # 初期回転なし
    objects.append({"mesh": mesh, "vertices": vertices, "pos": pos, "rotation": rotation})

# 移動範囲
bounds_min = np.array([-5.0, 0.0, -5.0])
bounds_max = np.array([5.0, 5.0, 5.0])

# Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)
for obj in objects:
    vis.add_geometry(obj["mesh"])

try:
    while True:
        for obj in objects:
            # ランダム移動
            delta = np.random.uniform(-0.3, 0.3, size=3)
            obj["pos"] += delta
            obj["pos"] = np.minimum(np.maximum(obj["pos"], bounds_min), bounds_max)

            # ランダム回転
            angles = np.radians(np.random.uniform(-5, 5, size=3))  # X,Y,Z回転角
            R = o3d.geometry.get_rotation_matrix_from_xyz(angles)
            obj["rotation"] = R @ obj["rotation"]

            # 頂点更新（回転＋位置）
            rotated = (obj["rotation"] @ vertices0.T).T
            obj["vertices"][:] = rotated + obj["pos"]
            obj["mesh"].vertices = o3d.utility.Vector3dVector(obj["vertices"])

        vis.update_geometry(None)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

except KeyboardInterrupt:
    pass

vis.destroy_window()