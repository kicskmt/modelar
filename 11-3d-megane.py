# ダウンロード先
# https://www.thingiverse.com/thing:2319951
# 眼鏡を描画。
# 角度をつけてみる

import cv2
import numpy as np
import open3d as o3d
import time


# ---------- STLをレンダリングする関数 ----------
def render_stl(stl_path, angle_x=0, angle_y=0, angle_z=0, size=600):
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.3, 0.6, 0.3])
    mesh.scale(2.0, center=mesh.get_center())

    # 3軸回転（X, Y, Z）をラジアンで指定
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

# ---------- 3パターンのヘルメット画像を事前に用意 ----------
stl_path = "glassesBody.stl"
helmet_imgs = {
    "left": render_stl(stl_path, angle_x=0, angle_y=-30, angle_z=0),
    "front": render_stl(stl_path, angle_x=0, angle_y=0, angle_z=0),
    "right": render_stl(stl_path, angle_x=0, angle_y=30, angle_z=0)
}

# ---------- 顔検出 ----------
print(cv2.data.haarcascades)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ---------- カメラ ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("カメラが見つかりません")


bak_x = 0
bak_y = 0
bak_w_face = 0
bak_h_face = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 顔を矩形で囲む
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    if len(faces) > 0:
        x, y, w_face, h_face = faces[0]
        print(x, y, w_face, h_face, bak_x - x, bak_y - y)

        print("x=",x, x-bak_x, (int)(x-bak_x)/20, (int)(bak_x + (x-bak_x)/20))
        print("y=",y, y-bak_y, (int)(y-bak_y)/20, (int)(bak_y + (y-bak_y)/20))
        x = (int)(bak_x + (x-bak_x)/20)
        y = (int)(bak_y + (y-bak_y)/20)
        print("x=",x, "  y=",y)
        w_face = (int)(bak_w_face + (w_face-bak_w_face)/20)
        h_face = (int)(bak_h_face + (h_face-bak_h_face)/20)

        # if bak_x - x < 30 and bak_x - x > -30 and bak_y-y <30 and bak_y-y > -30:
        if bak_y-y <30 and bak_y-y > -30:
        
            # 顔領域を左右に分けて明るさ比較 → 簡易的な横向き判定
            roi = gray[y:y+h_face, x:x+w_face]
            left_brightness = np.sum(roi[:, :w_face//2])
            right_brightness = np.sum(roi[:, w_face//2:])

            if right_brightness > left_brightness * 1.3:
                helmet_img = helmet_imgs["right"]
            elif left_brightness > right_brightness * 1.3:
                helmet_img = helmet_imgs["left"]
            else:
                helmet_img = helmet_imgs["front"]

            # ヘルメットを顔サイズに合わせてリサイズ
            scale = w_face / helmet_img.shape[1] * 1.3
            new_w = int(helmet_img.shape[1] * scale)
            new_h = int(helmet_img.shape[0] * scale)
            helmet_resized = cv2.resize(helmet_img, (new_w, new_h))

            # 合成位置（顔の上に少しずらす）
            x_offset = x - (new_w - w_face)*3 // 10
            y_offset = y - new_h*1 // 10
            # y_offset = y 

            # 合成範囲をフレーム内に収める
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
            # x, y, w_face, h_face
            bak_x = x
            bak_y = y
            bak_w_face = w_face
            bak_h_face = h_face



    cv2.imshow("Helmet Overlay", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()