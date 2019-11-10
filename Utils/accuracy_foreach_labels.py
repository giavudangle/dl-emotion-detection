import numpy as np

name_label = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def accuracy_labels(history_model, name):
    print("[INFO]: Accuracy for each labels")
    print("-" * 76)

    count_true_labels = history_model["Count_True_" + name]
    count_accuracy_labels = history_model["Count_Accuracy_" + name]

    # in kết quả
    for i in range(0, 7):
        print("[INFO]: Total {:10s} {:6d}, predict accuracy: {:6d}, accuracy = {:.2f}%".format(
            name_label[i].lower() + ":", count_true_labels[i], cot_accuracy_labels[i],
            count_accuracy_labels[i] * 100 / count_true_labels[i]))

    print("-" * 76)
    print(
        "[INFO]: Total {:10s} {:6d}, predict accuracy: {:6d}, accuracy = {:.2f}%".format("dataset:",
                                                                                         sum(count_true_labels),
                                                                                         sum(count_accuracy_labels),
                                                                                         sum(
                                                                                             count_accuracy_labels) * 100 / sum(
                                                                                             count_true_labels)))
