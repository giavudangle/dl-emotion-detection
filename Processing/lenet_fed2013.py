from Models.lenet import LeNet
from Utils import plotting, accuracy_foreach_labels, preprocessing_data
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import load_model
from sklearn.metrics import classification_report
import datetime, math
import keras
import os
import json
import numpy as np

# flag(0) => train, flag(1) => load
flag = 1
stt = 5
# 0 => val, 1 => test
val_or_test = 1

batch_size = 64
# có thể khởi tạo lr, decay ở def step_decay, nếu ko muốn dùng thì khởi tạo lr tại SGD
# opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
early_stopping = 50

mohinh = "LeNet"
datasetname = "fer2013"
depth = 1

image_size = 48
not_resize = True

epoch = 200
num_class = 7


class PrintLr(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        # If you want to apply decay.
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("lr: " + str(K.eval(lr_with_decay)))

        # # If you want to apply step_decay.
        # print("lr: " + str(K.eval(self.model.optimizer.lr)))


# learning rate schedule
def step_decay(epoch):
    # lr ban đầu
    initial_lrate = 0.1
    # giá trị drop sau 1 step, giảm 1 nữa lr sau 1 step
    drop = 0.5
    # số lượng epoch trong 1 step
    epochs_drop = 10.0
    # công thức tính
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


# đường dẫn model và thông tin biểu đồ model
save_path = "Output/" + mohinh + "/" + datasetname
if not os.path.exists(save_path):
    os.makedirs(save_path)
path_historymodel = save_path + "/model_history_(stt=" + str(stt) + ").json"
path_model = save_path + "/model_(stt=" + str(stt) + ").h5"

# train model
if flag != 1:
    # tiền xử lý dữ liệu
    train_Data, train_Labels, val_Data, val_Labels, test_Data, test_Labels = preprocessing_data.get_data(flag=flag,
                                                                                                         val_or_test=val_or_test,
                                                                                                         datasetname=datasetname,
                                                                                                         image_size=image_size,
                                                                                                         num_class=num_class,
                                                                                                         depth=depth,
                                                                                                         not_resize=not_resize)

    # thời điểm bắt đầu training
    hx, mx = int(str(datetime.datetime.now().time()).split(':')[-3]), int(
        str(datetime.datetime.now().time()).split(':')[-2])

    # build model
    print("[INFO]: Build model...")
    model = LeNet.build(numChannels=depth, imgRows=image_size, imgCols=image_size, numClasses=num_class)
    # compile model
    print("[INFO]: Compiling model...")
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # ### đưa các thông tin vào quá trình train
    # in lr
    # printlr = PrintLr()

    # đặt hệ số học lr
    # lrate = LearningRateScheduler(step_decay)

    # simple early stopping
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping)

    # save model
    modelcheckpoint = ModelCheckpoint(path_model, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    callbacks_list = [earlystopping, modelcheckpoint]

    # training model
    print("[INFO]: Training...")
    history = model.fit(train_Data, train_Labels, validation_data=(val_Data, val_Labels), batch_size=batch_size,
                        epochs=epoch, callbacks=callbacks_list, verbose=1)

    # thời điểm kết thúc training
    hy, my = int(str(datetime.datetime.now().time()).split(':')[-3]), int(
        str(datetime.datetime.now().time()).split(':')[-2])
    if hy < hx:
        hy += 24

    print("[INFO]: Save model...")
    # lưu thông tin model
    history_model = {"acc": history.history['acc'], "val_acc": history.history['val_acc'],
                     "loss": history.history['loss'], "val_loss": history.history['val_loss'],
                     "thoi_gian_train": (hy - hx) * 60 + (my - mx),
                     "batch_size": batch_size, "optimizer": str(opt), "image_size": image_size, "depth": depth}
    json.dump(history_model, open(path_historymodel, 'w'))

    print("-" * 76)
    # load best model in training
    print("[INFO]: Loading best model...")
    model = load_model(path_model)
    print("-" * 76)

# load history_model
history_model = json.loads(open(path_historymodel, 'r').read())

# xác định tập dữ liệu
if val_or_test < 1:
    eval_name = "Validation"
else:
    eval_name = "Test"

if not (eval_name + "_Accuracy" in history_model.keys()):
    if flag != 1:
        # load dữ liệu và nhãn
        if val_or_test < 1:
            X_Data = val_Data
            X_Labels = val_Labels
        else:
            X_Data = test_Data
            X_Labels = test_Labels
    else:
        # tiền xử lý dữ liệu
        X_Data, X_Labels = preprocessing_data.get_data(flag=flag, val_or_test=val_or_test, datasetname=datasetname,
                                                       image_size=image_size, num_class=num_class, depth=depth,
                                                       not_resize=not_resize)
        # load best model in training
        print("[INFO]: Loading best model...")
        model = load_model(path_model)
        print("-" * 76)

    # dự đoán kết quả của dữ liệu
    X_Predict = model.predict(X_Data)

    # tính toán loss và accuracy
    print("[INFO]: Evaluating in {}...".format(eval_name))
    (loss, accuracy) = model.evaluate(X_Data, X_Labels, batch_size=batch_size, verbose=1)

    # đếm số lượng ảnh của mỗi nhãn
    count_true_labels = [0, 0, 0, 0, 0, 0, 0]
    for i, true_label in enumerate(X_Labels):
        count_true_labels[int(np.argmax(true_label))] += 1

    # đếm số lượng nhãn chính xác của mỗi nhãn
    count_accuracy_labels = [0, 0, 0, 0, 0, 0, 0]
    for i, arr_predict_label in enumerate(X_Predict):
        true_label = int(np.argmax(X_Labels[i]))
        predict_label = int(np.argmax(arr_predict_label))
        if true_label == predict_label:
            count_accuracy_labels[true_label] += 1

    # lưu lại loss và accuracy
    accuracy = {eval_name + "_Loss": loss, eval_name + "_Accuracy": accuracy,
                eval_name + "_Predict": X_Predict.tolist(),
                eval_name + "_True": X_Labels.tolist(),
                "Count_Accuracy_" + eval_name: count_accuracy_labels,
                "Count_True_" + eval_name: count_true_labels}

    merged_json = {key: value for key, value in {**accuracy, **history_model}.items()}
    json.dump(merged_json, open(path_historymodel, 'w'))

    # load lại history_model
    history_model = json.loads(open(path_historymodel, 'r').read())
else:
    # load best model in training
    print("[INFO]: Loading best model...")
    model = load_model(path_model)
    print("-" * 76)

print("[INFO]: {} Accuracy: {:.2f}%".format(eval_name, history_model[eval_name + "_Accuracy"] * 100))
print("[INFO]: {} Loss: {}".format(eval_name, history_model[eval_name + "_Loss"]))

print("-" * 76)

tongm = history_model["thoi_gian_train"]

if tongm >= 60:
    print("[INFO]: Training time: " + str((int(tongm / 60))) + "h:" + str((int(tongm % 60))) + "p")
else:
    print("[INFO]: Training time: " + str(tongm) + "p")

print("[INFO]: Validation loss min in epoch: {}".format(
    str(history_model["val_loss"].index(min(history_model["val_loss"])) + 1)))

print("[INFO]: Training stop after: {} epoch".format(
    str(len(history_model["acc"]) - (history_model["val_loss"].index(min(history_model["val_loss"])) + 1))))

print("[INFO]: Training stop in epoch: {}".format(str(len(history_model["acc"]))))

print("[INFO]: Best model in epoch: {}".format(str(history_model["val_acc"].index(max(history_model["val_acc"])) + 1)))

print("-" * 76)
# tham số mô hình
print("[INFO]: Model parameters")
print("-" * 76)

print("[INFO]: Batch size: {}".format(str(history_model["batch_size"])))

print("[INFO]: Optimizer: {}".format(str(history_model["optimizer"])))

print("[INFO]: Image size: {}x{}".format(str(history_model["image_size"]), str(history_model["image_size"])))

print("[INFO]: Depth: {}".format(str(history_model["depth"])))

print("-" * 76)
# thông tin mô hình
print("[INFO]: Model summary")
model.summary()

# vẽ biểu đồ Accuracy, Loss, Confusion Matrix
plotting.plot_acc_or_loss(title='Training and Validation Accuracy', label_x='Accuracy', label_y='# Epoch',
                          train=history_model['acc'],
                          val=history_model['val_acc'])
plotting.plot_acc_or_loss(title='Training and Validation Loss', label_x='Loss', label_y='# Epoch',
                          train=history_model['loss'],
                          val=history_model['val_loss'])
plotting.plot_all(title='Training and Validation', label_x='Accuracy / Loss', label_y='# Epoch',
                  train_acc=history_model['acc'], train_loss=history_model['loss'],
                  val_acc=history_model['val_acc'], val_loss=history_model['val_loss'])
plotting.plot_confusion_matrix(y_true=np.array(history_model[eval_name + "_True"]),
                               y_pred=np.array(history_model[eval_name + "_Predict"]),
                               normalize=True, title='Confusion Matrix Normalized')
plotting.plot_confusion_matrix(y_true=np.array(history_model[eval_name + "_True"]),
                               y_pred=np.array(history_model[eval_name + "_Predict"]),
                               normalize=False, title='Confusion Matrix Not Normalized')
print("-" * 76)
# tính accuracy cho từng labels
accuracy_foreach_labels.accuracy_labels(history_model=history_model, name=eval_name)

# tính precision, recall, f1, support
print("-" * 76)
print(classification_report([np.argmax(x) for x in np.array(history_model[eval_name + "_True"])],
                            [np.argmax(x) for x in np.array(history_model[eval_name + "_Predict"])],
                            target_names=accuracy_foreach_labels.name_label))
