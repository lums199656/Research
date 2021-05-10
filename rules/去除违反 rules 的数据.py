# 第一列不能有 4
import pandas as pd
import numpy as np

TRUTH_PATH = "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult.csv"
SAVE_PATH = "./data_with_rule_single_rule.csv"


def single_cell_rule():
    file = pd.read_csv(TRUTH_PATH, sep=',')
    file = list(np.array(file))
    res = []
    for row in file:
        if row[1] == 4:
            row[1] = 6
        res.append(row)
    res = np.array(res)
    rsf_out = pd.DataFrame(res)
    rsf_out.to_csv(
        "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult_rule.csv",
        index=False, sep=',')
    print("输出完成！", res.shape)


def combine_rule():
    file = pd.read_csv(TRUTH_PATH, sep=',')
    labels = pd.read_csv(
        "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult_labels.csv",
        sep=',')
    labels_ = list(np.array(labels))
    file = list(np.array(file))
    positive = []
    labels = []
    negative = []
    col_1 = 0
    col_2 = 9
    for (i, _), label in zip(enumerate(file), labels_):
        is_true = True
        threshold = 0
        for j, _ in enumerate(file):
            if file[i][col_1] == file[j][col_1] and file[i][col_2] == file[j][col_2]:
                threshold -= 1
                if threshold <= 0:
                    is_true = False
                    negative.append(file[i])
                    labels.append([1])
                    break
        if is_true:
            positive.append(file[i])
            labels.append([0])
            # labels.append(label[0])
    X = np.array(positive + negative)
    # X = np.array(positive)
    y = np.array(labels)
    X_out = pd.DataFrame(X)
    y_out = pd.DataFrame(y)
    X_out.to_csv(
        "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult_rule.csv",
        index=False, sep=',')
    y_out.to_csv(
        "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult_rule_labels.csv",
        index=False, sep=',')

    # print(res)
    file_rule = pd.read_csv(
        "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult_rule.csv",
        sep=',')
    file = list(np.array(file_rule))
    error = 0
    for i, _ in enumerate(file):
        for j, _ in enumerate(file):
            if j <= i:
                continue
            if file[i][col_1] == file[j][col_1] and file[i][col_2] > file[j][col_2]:
                error += 1
    print(
        "输出完成！X_shape:{}, y_shape:{}, error: {}, error rate:{}".format(X.shape, y.shape, error, error / len(file) ** 2))


# single_cell_rule()
combine_rule()
