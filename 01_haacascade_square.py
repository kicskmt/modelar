
# 画像表示
# 顔画像認識の四角表示

import cv2

# 顔検出のためのHaarカスケード分類器を読み込む
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_cascade2 = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')



# Webカメラを開く
cap = cv2.VideoCapture(0)

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()
    
    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces2 = face_cascade2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    
    # 顔を矩形で囲む
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 顔を矩形で囲む
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