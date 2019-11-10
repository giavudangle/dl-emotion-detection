import cv2 as cv
import numpy as np
from keras.models import load_model

emotion_dict = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
colors = [(214, 38, 41), (140, 87, 74), (148, 102, 189), (43, 161, 43), (128, 128, 128),
          (255, 128, 13), (31, 120, 181)]

# fer2013, fer2013_6_dataset
mohinh = "VGG16"
datasetname = "fer2013_6_dataset"
stt = 1

image_resize = 48
depth = 1

# load model
path_model = "Output/" + mohinh + "/" + datasetname + "/model_(stt=" + str(stt) + ").h5"
model = load_model(path_model)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # đọc ảnh xám
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    # nhận diện khuôn mặt
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    # tỉ lệ scale=1.3, lấy tối đa 5 khuôn mặt
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # lấy khuông mặt
        face_gray = gray[y:y + h, x:x + w]

        # # show ảnh khuôn mặt
        # cv.imshow("face", face_gray)

        if depth == 3:
            # chuyển sang 3 gam màu
            image_color = cv.cvtColor(face_gray, cv.COLOR_GRAY2RGB)
            # resize ảnh sang h x w, mở rộng mảng(có giá trị là (h, w, d)) ở phía đầu(tại vị trí 0)==>(1, h, w, d)
            cropped_img = np.expand_dims(cv.resize(image_color, (image_resize, image_resize)), 0)
        else:
            # ảnh 1 gam màu
            # resize ảnh sang h x w, mở rộng mảng(có giá trị là (h, w)) sang ảnh có giá trị==>(1, h, w, 1)
            cropped_img = np.expand_dims(np.expand_dims(cv.resize(face_gray, (image_resize, image_resize)), 0), -1)

        # normalize ảnh
        cv.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv.NORM_L2, dtype=cv.CV_32F)
        cropped_img = cropped_img.astype("float32") / 255.0

        # dự đoán ảnh trên model
        prediction = model.predict(cropped_img)

        # vẽ khung bao quanh khuôn mặt
        cv.rectangle(frame, (x, y), (x + w, y + h), colors[int(np.argmax(prediction))], 2)

        # show nhãn tương ứng
        cv.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   colors[int(np.argmax(prediction))],
                   1, cv.LINE_AA)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
