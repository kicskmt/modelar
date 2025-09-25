
# ウィッチハットのドロップ跳ね返り
# 跳ね返り時の色変更
# オブジェクトの数変更
# zball1.stlを読み込んでボールにする


import open3d as o3d
import numpy as np
import time
import random
import copy

# STL読み込み
# base_mesh = o3d.io.read_triangle_mesh("zWitch_Hat.stl")
base_mesh = o3d.io.read_triangle_mesh("zcrystal.stl")
base_mesh.compute_vertex_normals()
base_mesh.paint_uniform_color([1.0, 0.84, 0.0])
vertices0 = np.asarray(base_mesh.vertices)
center = vertices0.mean(axis=0)
vertices0 -= center

# 複数オブジェクト
num_objects = 3    ##数を変えてみる
objects = []

for _ in range(num_objects):
    mesh = copy.deepcopy(base_mesh)
    vertices = vertices0.copy()
    # 上空から降ってくるイメージでYを大きめに設定
    pos = np.array([
        random.uniform(-30.0, 100.0),
        random.uniform(30.0, 30.0),#高さ100にすると少し高くなる
        random.uniform(-30.0, 100.0)
    ])
    # pos = np.array([
    #     random.uniform(-30.0, 30.0),
    #     random.uniform(150.0, 50.0),
    #     random.uniform(-30.0, 30.0)
    # ])
    vel = np.array([random.uniform(-0.1,0.1), 0.0, random.uniform(-0.1,0.1)])
    rotation = np.eye(3)
    angular_vel = np.radians(np.random.uniform(-20, 20, size=3))
    objects.append({
        "mesh": mesh, "vertices": vertices, "pos": pos,
        "vel": vel, "rotation": rotation, "angular_vel": angular_vel
    })

# 移動範囲
bounds_min = np.array([-5.0, 0.0, -5.0])
bounds_max = np.array([5.0, 15.0, 5.0])  # 上限を長めに
bounds_max = np.array([5.0, 100.0, 5.0])  # 上限を長めに

# gravity = -9.8
# dt = 0.02
# bounce = -0.8  # Y方向跳ね返り係数
# friction = 0.98  # X/Z方向微小減衰
gravity = -19.8
dt = 0.05
bounce = -1.0  # Y方向跳ね返り係数
friction = 1.0  # X/Z方向微小減衰

vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)
for obj in objects:
    vis.add_geometry(obj["mesh"])

# ランダムな色を作る関数
def random_color():
    return [random.random(), random.random(), random.random()]



try:
    while True:
        for obj in objects:
            # 重力
            obj["vel"][1] += gravity * dt
            # 位置更新
            obj["pos"] += obj["vel"] * dt

            # X/Z微動（摩擦）
            obj["vel"][0] *= friction
            obj["vel"][2] *= friction

            # Y軸常に跳ね返る
            if obj["pos"][1] < bounds_min[1]:
                obj["pos"][1] = bounds_min[1]
                obj["vel"][1] *= bounce
                # mesh.paint_uniform_color(random_color())
                # obj["mesh"].paint_uniform_color(random_color())
                
                if abs(obj["vel"][1]) < 0.1:  # 小さい速度でも必ず跳ね返す
                    obj["vel"][1] = 2.0

            # X/Z方向範囲制限
            for i in [0,2]:
                if obj["pos"][i] < bounds_min[i]:
                    obj["pos"][i] = bounds_min[i]
                    obj["vel"][i] *= -0.5
                if obj["pos"][i] > bounds_max[i]:
                    obj["pos"][i] = bounds_max[i]
                    obj["vel"][i] *= -0.5

            # 回転更新
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