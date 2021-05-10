import pandas as pd
import numpy as np

TRUTH_PATH = "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult.csv"
RULE_PATH = "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult_rule.csv"
SAMPLE_IN_TAGN_FOLDER = "/Users/luminshen/Documents/代码/PycharmProjects/Research/-GAN-/Table-GAN/tableGAN/data/Adult/Adult_rule.csv"
# SAMPLE_NEW_GAN_FOLDER = "/Users/luminshen/Desktop/模型/new gan/600分钟 w2/Adult_OI_11_00_fake.csv"
SAMPLE_NEW_GAN_FOLDER = "/Users/luminshen/Desktop/模型/new gan/0相同，10相同/600分钟 w2/Adult_OI_11_00_fake.csv"

SAMPLE_ORIGINAL_FOLDER = "/Users/luminshen/Desktop/模型/original gan/600分钟/Adult_OI_11_00_fake.csv"

def verify_single_cell_rule(file):
    file = list(np.array(file))
    error = 0
    for row in file:
        if row[1] == 1:
            error += 1

    print("error: {}, error rate:{}".format(error, error / len(file)))


def verify_combine_rule(file):
    file = list(np.array(file))
    error = 0
    col_1 = 6
    col_2 = 10
    count = 0
    for i, _ in enumerate(file):
        isOk = True
        for j, _ in enumerate(file):
            if j < i:
                continue
            if file[i][col_1] > file[j][col_1] and file[i][col_2] > file[j][col_2]:
                error += 1
                isOk = False
        if isOk:
            count += 1
    print("count: {}, error: {}, error rate:{}".format(count, error, error / len(file) ** 2))


file_real = pd.read_csv(TRUTH_PATH, sep=',')
file_rule = pd.read_csv(RULE_PATH, sep=',')
file_fake = pd.read_csv(SAMPLE_IN_TAGN_FOLDER, sep=';')
file_fake_new = pd.read_csv(SAMPLE_NEW_GAN_FOLDER, sep=';')
file_fake_original = pd.read_csv(SAMPLE_ORIGINAL_FOLDER, sep=';')
print("real----")
# verify_single_cell_rule(file_real)
verify_combine_rule(file_real)
# print("rule----")
# verify_combine_rule(file_rule)
# print("inside----")
# # verify_single_cell_rule(file_fake)
# verify_combine_rule(file_fake)
print("new----")
# verify_single_cell_rule(file_fake_new)
verify_combine_rule(file_fake_new)
print("original----")
# verify_single_cell_rule(file_fake_original)
verify_combine_rule(file_fake_original)
