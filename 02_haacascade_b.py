
# 顔検出後の帽子画像の表示
# 帽子の位置の調整

import cv2
import numpy as np

# カスケード分類器の読み込み
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 帽子画像（透過PNG）を読み込む
glasses = cv2.imread('obj\\xbousi.png', cv2.IMREAD_UNCHANGED)  # 4チャンネル（RGBA）で読み込む

# アルファ付き画像をBGRとアルファに分離
def overlay_image(bg, fg, x, y):
    h, w = fg.shape[:2]

    # 背景のサイズ
    bh, bw = bg.shape[:2]

    # 範囲をクリッピング
    if x < 0:
        fg = fg[:, -x:]       # 左がはみ出た分を切る
        w = fg.shape[1]
        x = 0
    if y < 0:
        fg = fg[-y:, :]       # 上がはみ出た分を切る
        h = fg.shape[0]
        y = 0
    if x + w > bw:
        fg = fg[:, :bw - x]   # 右がはみ出た分を切る
        w = fg.shape[1]
    if y + h > bh:
        fg = fg[:bh - y, :]   # 下がはみ出た分を切る
        h = fg.shape[0]

    # 透過マスク
    alpha_fg = fg[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_fg

    for c in range(3):
        bg[y:y+h, x:x+w, c] = (alpha_fg * fg[:, :, c] + alpha_bg * bg[y:y+h, x:x+w, c])
    
    return bg

# カメラ起動
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換して顔検出
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        # 画像のサイズを顔に合わせてリサイズ
        # resized_glasses = cv2.resize(glasses, (w, int(w/2)))
        resized_glasses = cv2.resize(glasses, (w, int(w/3)))

        # 顔の上あたりに配置
        # gy = y - int(h / 4)
        gy = y - int(h / 10)*1
        
        frame = overlay_image(frame, resized_glasses, x, gy)

    cv2.imshow('Glasses Overlay', frame)
    if cv2.waitKey(1) == 27:  # ESCキーで終了
        break

cap.release()
cv2.destroyAllWindows()