from matplotlib import pyplot as plt
import itertools
import numpy as np

classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def my_confusion_matrix(y_true, y_pred):
    N = y_true.shape[1]
    cm = np.zeros([N, N], dtype=int)
    for n in range(y_true.shape[0]):
        cm[int(np.argmax(y_true[n])), int(np.argmax(y_pred[n]))] += 1
    return cm


def plot_confusion_matrix(y_true, y_pred, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = my_confusion_matrix(y_true=y_true, y_pred=y_pred)

    plt.subplots(1, 1, figsize=(10, 10))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_acc_or_loss(title, label_x, label_y, train, val):
    # quy định màu và nét vẽ
    if 'Accuracy' == label_x:
        x = 'bo-'
        y = 'r*-'
    else:
        x = 'gx-'
        y = 'ys-'

    plt.subplots(1, 1, figsize=(12, 8))
    plt.title(title, fontsize=22)
    # epoch bắt đầu từ 1 tới len(train)
    plt.xlim(1, len(train))
    # np.append(np.roll(train, 1), train[len(train) - 1] chuyển mảng bắt đầu từ 0 sang bắt đầu từ 1
    plt.plot(np.append(np.roll(train, 1), train[len(train) - 1]), x, label='Training_' + label_x)
    plt.plot(np.append(np.roll(val, 1), val[len(val) - 1]), y, label='Validation_' + label_x)
    plt.xlabel(label_y, fontsize=20)
    plt.ylabel(label_x, fontsize=20)
    plt.grid()
    plt.legend(fontsize=15)
    plt.show()


def plot_all(title, label_x, label_y, train_acc, val_acc, train_loss, val_loss):
    plt.subplots(1, 1, figsize=(12, 8))
    plt.title(title, fontsize=22)
    # epoch bắt đầu từ 1 tới len(train)
    plt.xlim(1, len(train_acc))
    # np.append(np.roll(train, 1), train[len(train) - 1] chuyển mảng bắt đầu từ 0 sang bắt đầu từ 1
    plt.plot(np.append(np.roll(train_acc, 1), train_acc[len(train_acc) - 1]), 'bo-', label='Training_Accuracy')
    plt.plot(np.append(np.roll(train_loss, 1), train_loss[len(train_loss) - 1]), 'gx-', label='Training_Loss')
    plt.plot(np.append(np.roll(val_acc, 1), val_acc[len(val_acc) - 1]), 'r*-', label='Validation_Accuracy')
    plt.plot(np.append(np.roll(val_loss, 1), val_loss[len(val_loss) - 1]), 'ys-', label='Validation_Loss')
    plt.xlabel(label_y, fontsize=20)
    plt.ylabel(label_x, fontsize=20)
    plt.grid()
    plt.legend(fontsize=15)
    plt.show()
