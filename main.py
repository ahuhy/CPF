import logging
import numpy as np
import pandas
from load_data import DATA
from LPKT import LPKT
import pandas as pd
import torch


def generate_q_matrix(path, n_skill, n_problem, gamma=0.0):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            problem2skill = eval(line)
    q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
    for p in problem2skill.keys():
        q_matrix[p][problem2skill[p]] = 1
    return q_matrix


def generate_p_matrix(file_path, p_gamma):
    with open(file_path, 'r') as f:
        data_str = f.read()
    data = eval(data_str)
    max_key = max(data.keys())
    max_len = max([len(data[key]) for key in data])
    matrix = np.zeros((max_key + 1, max_len), dtype=float)
    for key in data:
        for i, val in enumerate(data[key]):
            if val != -1:
                matrix[key][i] = val ** p_gamma 

    return matrix

""" # k-fold cross validation
k, train_auc_sum, valid_auc_sum = 5, .0, .0
for i in range(k):
    lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix,  batch_size, dropout)
    train_data = dat.load_data('../../data/anonymized_full_release_competition_dataset/train' + str(i) + '.txt')
    valid_data = dat.load_data('../../data/anonymized_full_release_competition_dataset/valid' + str(i) + '.txt')
    best_train_auc, best_valid_auc = lpkt.train(train_data, valid_data, epoch=30, lr=0.003, lr_decay_step=10)
    print('fold %d, train auc %f, valid auc %f' % (i, best_train_auc, best_valid_auc))
    train_auc_sum += best_train_auc
    valid_auc_sum += best_valid_auc
print('%d-fold validation: avg of best train auc %f, avg of best valid auc %f' % (k, train_auc_sum / k, valid_auc_sum / k)) """

batch_size = 32
n_at = 1326
n_it = 2839
n_question = 102
n_exercise = 3162
seqlen = 500
d_k = 128
d_a = 50
d_e = 128
d_f = 3162
d_l = 1709
d_x = 3000
q_gamma = 0.03
p_gamma = 0.03
dropout = 0.2 

q_matrix = generate_q_matrix(
    './data/anonymized_full_release_competition_dataset/problem2skill',
    n_question, n_exercise,
    q_gamma
)

p_matrix = generate_p_matrix('./CPF/dependency_dict_extend_pp_assist2017.txt', p_gamma)

dat = DATA(seqlen=seqlen, separate_char=',')
logging.getLogger().setLevel(logging.INFO)


train_data = dat.load_data('./data/assist2017train_va_te/train0.txt')
valid_data = dat.load_data('./data/assist2017train_va_te/valid0.txt')
test_data = dat.load_data('./data/assist2017train_va_te/test.txt')

lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, d_l, d_f, q_matrix, p_matrix, batch_size, dropout)
lpkt.train(train_data, valid_data, epoch=30, lr=0.003, lr_decay_step=10)
lpkt.load("./CPF/2017params/lpkt3.params")
_, auc, accuracy, rmse, r2 = lpkt.eval(test_data)
print("auc: %.6f, accuracy: %.6f,  rmse: %.6f,  r2: %.6f" % (auc, accuracy, rmse, r2))

""" _, auc, accuracy, r2, rmse = lpkt.eval(test_data)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
print("r2: %.6f, rmse: %.6f" % (r2, rmse)) """

