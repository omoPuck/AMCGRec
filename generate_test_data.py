import pickle as pkl
import numpy as np
from sklearn.metrics import ndcg_score

training_matrix_f = './data/adjacency_matrix.p'
ground_truth_matrix_f = './data/user_action.p'

with open(training_matrix_f, 'rb') as f:
    training_matrix = pkl.load(f)
    if not isinstance(training_matrix, np.matrix):
        training_matrix = training_matrix.todense()
    else:
        training_matrix = np.array(training_matrix)

with open(ground_truth_matrix_f, 'rb') as f:
    ground_truth_matrix = pkl.load(f)
    if not isinstance(ground_truth_matrix, np.matrix):
        ground_truth_matrix = ground_truth_matrix.todense()
    else:
        ground_truth_matrix = np.array(ground_truth_matrix)
user_size = training_matrix.shape[0]
# item_size = training_matrix.shape[1]

negativate = []
for i in range(user_size):
    neg_indices = np.where(ground_truth_matrix[i] == 0)[0]
    training_indices = np.where(training_matrix[i] > 0)[0]
    testing_indices = [x for x in np.where(ground_truth_matrix[i] > 0)[0] if x not in training_indices]
    for test_ in testing_indices:
        neg_ = np.random.choice(np.array(neg_indices), 99, replace=False)
        addpos_ = np.append(neg_,test_)
        negativate_ = np.concatenate((np.zeros((100,1))+ i,addpos_.reshape(-1,1)),axis=1)
        negativate.append(negativate_)

test_negative = np.array(negativate)

np.save("./data/test_negative.npy",test_negative)