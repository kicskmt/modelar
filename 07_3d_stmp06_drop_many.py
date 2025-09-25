import open3d as o3d
import numpy as np
import time
import random
import copy

# STL読み込み
base_mesh = o3d.io.read_triangle_mesh("zWitch_Hat.stl")
base_mesh.compute_vertex_normals()
base_mesh.paint_uniform_color([1.0, 0.84, 0.0])

# 小さくスケーリング
scale_factor = 0.05
base_mesh.scale(scale_factor, center=base_mesh.get_center())

vertices0 = np.asarray(base_mesh.vertices)
center = vertices0.mean(axis=0)
vertices0 -= center

# 複数オブジェクト
num_objects = 50
objects = []

for _ in range(num_objects):
    mesh = copy.deepcopy(base_mesh)
    vertices = vertices0.copy()
    pos = np.array([
        random.uniform(-3.0, 3.0),
        random.uniform(5.0, 10.0),
        random.uniform(-3.0, 3.0)
    ])
    vel = np.array([random.uniform(-0.1,0.1), 0.0, random.uniform(-0.1,0.1)])
    rotation = np.eye(3)
    angular_vel = np.radians(np.random.uniform(-20, 20, size=3))
    objects.append({
        "mesh": mesh, "vertices": vertices, "pos": pos,
        "vel": vel, "rotation": rotation, "angular_vel": angular_vel
    })

# 移動範囲
bounds_min = np.array([-5.0, 0.0, -5.0])
bounds_max = np.array([5.0, 15.0, 5.0])

gravity = -9.8
dt = 0.02
bounce = -0.9
friction = 0.98

# Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)

# カメラを少し引く
# ctr = vis.get_view_control()
# ctr.set_up([0, 1, 0])
# ctr.set_front([0, -0.3, -1])
# ctr.set_lookat([0, 5, 0])
# ctr.set_zoom(0.7)  # ズームアウト

# カメラ設定（Visualizer 作成後に追加）
# ctr = vis.get_view_control()
# ctr.set_up([0, 1, 0])                # 上方向はY軸
# ctr.set_front([0, -0.5, -1])          # カメラの向き（斜め下）
# ctr.set_lookat([0, 5, 0])             # 注視点
# ctr.set_zoom(0.6)                     # ズームアウト（0.6→0.5でさらに引くことも可能）

# ctr = vis.get_view_control()
# ctr.set_up([0, 1, 0])                # 上方向はY軸
# ctr.set_front([0, 0.3, -1])          # カメラの向き（斜め下）
# ctr.set_lookat([0, 500, 0])             # 注視点
# ctr.set_zoom(1000.01)                     # ズームアウト（0.6→0.5でさらに引くことも可能）


ctr = vis.get_view_control()
ctr.set_up([0, 1, 0])           # 上方向
ctr.set_front([0, -0.7, -2.5])  # カメラ位置をさらに後方に
ctr.set_lookat([0, 5, 0])       # 注視点を落下範囲の中心
ctr.set_zoom(0.6)    

for obj in objects:
    vis.add_geometry(obj["mesh"])

try:
    while True:
        for obj in objects:
            # 重力
            obj["vel"][1] += gravity * dt
            obj["pos"] += obj["vel"] * dt

            # X/Z摩擦
            obj["vel"][0] *= friction
            obj["vel"][2] *= friction

            # Y軸常に跳ね返る
            if obj["pos"][1] < bounds_min[1]:
                obj["pos"][1] = bounds_min[1]
                obj["vel"][1] *= bounce
                if abs(obj["vel"][1]) < 0.1:
                    obj["vel"][1] = 2.0

            # X/Z範囲制限
            for i in [0,2]:
                if obj["pos"][i] < bounds_min[i]:
                    obj["pos"][i] = bounds_min[i]
                    obj["vel"][i] *= -0.5
                if obj["pos"][i] > bounds_max[i]:
                    obj["pos"][i] = bounds_max[i]
                    obj["vel"][i] *= -0.5

            # 回転
            angles = obj["angular_vel"] * dt
            R = o3d.geometry.get_rotation_matrix_from_xyz(angles)
            obj["rotation"] = R @ obj["rotation"]

            # 頂点更新
            rotated = (obj["rotation"] @ vertices0.T).T
            obj["vertices"][:] = rotated + obj["pos"]
            obj["mesh"].vertices = o3d.utility.Vector3dVector(obj["vertices"])

        vis.update_geometry(None)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(dt)

except KeyboardInterrupt:
    pass

vis.destroy_window()