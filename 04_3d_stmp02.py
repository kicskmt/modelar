
# 最初に帽子の表示
# 次に、起動時帽子のカラフルな表示
# 毎秒ごとにカラフルな表示



import open3d as o3d
import numpy as np
import time
import random

# ランダムな色を作る関数
def random_color():
    return [random.random(), random.random(), random.random()]

# メイン
mesh = o3d.io.read_triangle_mesh("zWitch_Hat.stl")
mesh.compute_vertex_normals()

# 初期色は金色
mesh.paint_uniform_color([1.0, 0.84, 0.0])

# Visualizer の作成
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="STL Viewer", width=800, height=600)
vis.add_geometry(mesh)

# mesh.paint_uniform_color(random_color())

# 更新ループ
try:
    while True:
        # 色を毎フレーム変更
        # mesh.paint_uniform_color(random_color())
        
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        
        time.sleep(0.1)  # 0.1秒ごとに色を変える
except KeyboardInterrupt:
    pass

vis.destroy_window()