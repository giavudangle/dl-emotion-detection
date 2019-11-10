from Utils import preprocessing_data
from keras.models import load_model
import numpy as np
import gc

image_size = 48
depth = 1
not_resize = True
num_class = 7

# val = 0 test = 1
val_or_test = 0

datasetname = "fer2013"

# đường dẫn model và thông tin biểu đồ model
# model 1
mohinh_1 = "VGG16"
stt_1 = 3
path_model_1 = "Output/" + mohinh_1 + "/" + datasetname
load_path_model_1 = path_model_1 + "/model_(stt=" + str(stt_1) + ").h5"

# model 2
mohinh_2 = "VGG19"
stt_2 = 1
path_model_2 = "Output/" + mohinh_2 + "/" + datasetname
load_path_model_2 = path_model_2 + "/model_(stt=" + str(stt_2) + ").h5"

# model 3
mohinh_3 = "VGG13"
stt_3 = 1
path_model_3 = "Output/" + mohinh_3 + "/" + datasetname
load_path_model_3 = path_model_3 + "/model_(stt=" + str(stt_3) + ").h5"

# model 3
mohinh_4 = "LeNet"
stt_4 = 5
path_model_4 = "Output/" + mohinh_4 + "/" + datasetname
load_path_model_4 = path_model_4 + "/model_(stt=" + str(stt_4) + ").h5"

# load models
print("[INFO]: Loading model 1...")
model_1 = load_model(load_path_model_1)
print("-" * 76)
print("[INFO]: Loading model 2...")
model_2 = load_model(load_path_model_2)
print("-" * 76)
print("[INFO]: Loading model 3...")
model_3 = load_model(load_path_model_3)
print("-" * 76)
print("[INFO]: Loading model 4...")
model_4 = load_model(load_path_model_4)
print("-" * 76)

# lấy dữ liệu
X_Data, X_Labels = preprocessing_data.get_data(flag=1, val_or_test=val_or_test, datasetname=datasetname,
                                               image_size=image_size,
                                               num_class=num_class, depth=depth,
                                               not_resize=not_resize)

print("[INFO]: Predict...")
print("-" * 76)
# X_Predict
size_data = len(X_Data)
X_Predict = []
for j, image_data in enumerate(X_Data):
    # dự đoán kết quả của dữ liệu
    accuracy_image_1 = model_1.predict(np.expand_dims(image_data, 0))
    accuracy_image_2 = model_2.predict(np.expand_dims(image_data, 0))
    accuracy_image_3 = model_3.predict(np.expand_dims(image_data, 0))
    accuracy_image_4 = model_4.predict(np.expand_dims(image_data, 0))

    id_result = -1
    best_accuracy = -1
    for i in range(7):
        x = (accuracy_image_1[0][i] + accuracy_image_2[0][i] + accuracy_image_3[0][i] + accuracy_image_4[0][i]) / 4
        if best_accuracy < x:
            best_accuracy = x
            id_result = i

    X_Predict.append(id_result)
    if (j + 1) % 100 == 0:
        print("[INFO]: Images predict: {}/{}".format(j + 1, size_data))
        gc.collect()

# đếm số lượng ảnh của mỗi nhãn
count_true_labels = [0, 0, 0, 0, 0, 0, 0]
for i, true_label in enumerate(X_Labels):
    count_true_labels[int(np.argmax(true_label))] += 1

# đếm số lượng nhãn chính xác của mỗi nhãn
count_accuracy_labels = [0, 0, 0, 0, 0, 0, 0]
for i, predict_label in enumerate(X_Predict):
    true_label = int(np.argmax(X_Labels[i]))
    if true_label == predict_label:
        count_accuracy_labels[true_label] += 1

# accuracy for each labels
name_label = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

print("[INFO]: Accuracy for each labels")
print("-" * 76)

# in kết quả
for i in range(0, 7):
    print("[INFO]: Total {:10s} {:6d}, predict accuracy: {:6d}, accuracy = {:.2f}%".format(name_label[i].lower() + ":",
                                                                                           count_true_labels[i],
                                                                                           count_accuracy_labels[i],
                                                                                           count_accuracy_labels[
                                                                                               i] * 100 /
                                                                                           count_true_labels[i]))
print("-" * 76)
print(
    "[INFO]: Total {:10s} {:6d}, predict accuracy: {:6d}, accuracy = {:.2f}%".format("dataset:", sum(count_true_labels),
                                                                                     sum(count_accuracy_labels), sum(
            count_accuracy_labels) * 100 / sum(count_true_labels)))
