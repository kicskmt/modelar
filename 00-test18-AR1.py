import cv2
import numpy as np
import open3d as o3d

# === STLファイル設定 ===
stl_path = "zWitch_Hat.stl"
a_x, a_y, a_z = 90, 180, 180
bairitu = 2.0  # 拡大率

# === STLを画像にレンダリングする関数 ===
def render_stl(stl_path, angle_x=0, angle_y=0, angle_z=0, size=300):
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1.0, 0.5, 0.0])  # オレンジ色
    mesh.scale(2.0, center=mesh.get_center())

    # 回転
    R = mesh.get_rotation_matrix_from_xyz((
        np.radians(angle_x),
        np.radians(angle_y),
        np.radians(angle_z)
    ))
    mesh.rotate(R, center=mesh.get_center())

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=size, height=size)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 1, 0])  # 緑背景
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)

    # 背景緑を透過にする
    lower = np.array([0, 250, 0, 0], dtype=np.uint8)
    upper = np.array([5, 255, 5, 255], dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)
    img[:, :, 3] = cv2.bitwise_not(mask)

    return img


# === 事前にSTLをレンダリング（正面表示だけ用意） ===
helmet_img = render_stl(stl_path, angle_x=a_x, angle_y=a_y, angle_z=a_z)


# === カメラ準備 ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("カメラが見つかりません")


# === ArUcoマーカ検出の準備 ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # マーカ検出
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        for corner, marker_id in zip(corners, ids):
            pts = corner[0].astype(int)

            # マーカの中心を計算
            x_center = int(np.mean(pts[:, 0]))
            y_center = int(np.mean(pts[:, 1]))
            marker_size = int(np.linalg.norm(pts[0] - pts[2]))  # 対角長さをスケールに利用

            # モデルのリサイズ
            scale = marker_size / helmet_img.shape[1] * bairitu
            new_w = int(helmet_img.shape[1] * scale)
            new_h = int(helmet_img.shape[0] * scale)
            helmet_resized = cv2.resize(helmet_img, (new_w, new_h))

            # 合成位置（マーカの中心に配置）
            x_offset = x_center - new_w // 2
            y_offset = y_center - new_h // 2

            # 画面外チェック
            y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + new_h)
            x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + new_w)

            helmet_y1, helmet_y2 = max(0, -y_offset), new_h - max(0, (y_offset + new_h) - frame.shape[0])
            helmet_x1, helmet_x2 = max(0, -x_offset), new_w - max(0, (x_offset + new_w) - frame.shape[1])

            # アルファブレンド
            alpha_helmet = helmet_resized[helmet_y1:helmet_y2, helmet_x1:helmet_x2, 3] / 255.0
            alpha_frame = 1.0 - alpha_helmet
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (alpha_frame * frame[y1:y2, x1:x2, c] +
                                          alpha_helmet * helmet_resized[helmet_y1:helmet_y2, helmet_x1:helmet_x2, c])

            # マーカの枠を描画（確認用）
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    cv2.imshow("AR Marker + 3D Model", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESCで終了
        break

cap.release()
cv2.destroyAllWindows()