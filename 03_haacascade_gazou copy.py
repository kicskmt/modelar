
# 顔検出後の頭、眼鏡、口部分の表示

import cv2
from PIL import Image

file_glasses = 'xbousi.png'
file_megane = 'xmegane.png'
file_mouth = 'xmouth.png'

# 顔検出のためのHaarカスケード分類器を読み込む
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# メガネ画像（透過PNG）を読み込む
glasses = cv2.imread('obj\\xbousi.png', cv2.IMREAD_UNCHANGED)  # 4チャンネル（RGBA）で読み込む
megane = cv2.imread('obj\\xmegane.png', cv2.IMREAD_UNCHANGED)  # 4チャンネル（RGBA）で読み込む
mouth = cv2.imread('obj\\xmouth.png', cv2.IMREAD_UNCHANGED)  # 4チャンネル（RGBA）で読み込む


# 入力ファイル名と出力ファイル名
input_file = 'obj\\xkatura2.png'
output_file = 'obj\\xkatura.png'




def toumeika(input_file, output_file):
    # 画像をRGBA（アルファ付き）で読み込み
    img = Image.open(input_file).convert("RGBA")
    datas = img.getdata()

    new_data = []
    for item in datas:
        # itemは(R, G, B, A)
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    img.putdata(new_data)
    img.save(output_file, "PNG")

    print("白色を透明に変換して保存しました:", output_file)



toumeika(input_file, output_file)
katura = cv2.imread('obj\\xkatura.png', cv2.IMREAD_UNCHANGED)  # 4チャンネル（RGBA）で読み込む



def overlay_image(bg, fg, x, y):
    h, w = fg.shape[:2]

    # 背景画像サイズ
    bg_h, bg_w = bg.shape[:2]

    # はみ出しチェック
    if x < 0 or y < 0 or x + w > bg_w or y + h > bg_h:
        return bg  # 合成せずそのまま返す

    # 透過マスクの生成
    alpha_fg = fg[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_fg

    for c in range(3):  # B, G, R チャンネル
        bg[y:y+h, x:x+w, c] = (alpha_fg * fg[:, :, c] + alpha_bg * bg[y:y+h, x:x+w, c])

    return bg


# Webカメラを開く
cap = cv2.VideoCapture(0)

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()
    
    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    
    # 顔を矩形で囲む
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # resized_glasses = cv2.resize(glasses, (w, int(h/3)))
        # # 顔の上1/3あたりに配置
        # # gy = y + int(h / 4)
        # gy = y - int(h / 4)
        # frame = overlay_image(frame, resized_glasses, x, gy)

        resized_megane = cv2.resize(megane, (w, int(h/3)))
        # 顔の上1/3あたりに配置
        gy2 = y + int(h / 4)
        frame = overlay_image(frame, resized_megane, x, gy2)

        # 口
        resized_mouth = cv2.resize(mouth, (w, int(h/3)))
        gy2 = y + int(h*3 / 4)
        frame = overlay_image(frame, resized_mouth, x, gy2)

        resized_katura = cv2.resize(katura, (w, int(h/3)))
        # 顔の上1/3あたりに配置
        # gy = y + int(h / 4)
        gy = y - int(h / 4)
        frame = overlay_image(frame, resized_katura, x, gy)


    # 目を矩形で囲む
    # faces2 = face_cascade2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # for (x, y, w, h) in faces2:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # 結果を表示
    cv2.imshow('Face Detection', frame)
    
    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Webカメラを解放
cap.release()
cv2.destroyAllWindows()





