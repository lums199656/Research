# rule 为 0 大时，9 也要大
import numpy as np
import pandas as pd

# 1. GT 的 error 是多少
TRUTH_PATH = "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult.csv"
TRUTH_LABEL_PATH = "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult_labels.csv"
COL_1 = 0
COL_2 = 9


def verify_rule():
    file = pd.read_csv(TRUTH_PATH, sep=',')
    file = list(np.array(file))
    error = 0
    col_1 = COL_1
    col_2 = COL_2
    count = 0
    for i, _ in enumerate(file):
        for j, _ in enumerate(file):
            if file[i][col_1] > file[j][col_1] and file[i][col_2] > file[j][col_2]:
                error += 1
            count += 1
    print("GT: length:{} error: {}, error rate:{}".format(count, error, error / len(file) ** 2))


verify_rule()

# 2. 生成数据
X_OUTPUT = "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult_pairwise.csv"
Y_OUTPUT = "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult_pairwise_labels.csv"


def generate_pairwise_data():
    file = list(np.array(pd.read_csv(TRUTH_PATH, sep=',')))
    positive = []
    labels = []
    negative = []
    col_1 = COL_1
    col_2 = COL_2
    error = 0
    for i, _ in enumerate(file):
        for j, _ in enumerate(file):
            if file[i][col_1] > file[j][col_1] and file[i][col_2] > file[j][col_2]:
                negative.append(file[i] - file[j])
                labels.append([1])
                error += 1
            else:
                positive.append(file[i] - file[j])
                labels.append([0])

    X = np.array(positive + negative)
    y = np.array(labels)
    X_out = pd.DataFrame(X)
    y_out = pd.DataFrame(y)
    X_out.to_csv(X_OUTPUT, index=False, sep=',')
    y_out.to_csv(Y_OUTPUT, index=False, sep=',')
    print("输出完成！Xshape:{}, yshape:{} error: {}, error rate:{}".format(X.shape, y.shape, error, error / len(X)))


generate_pairwise_data()

