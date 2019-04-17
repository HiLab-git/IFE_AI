import numpy as np
import pandas as pd

ground_truth_files = ['/home/guotai/data/phrosis_data/names/old_fold_5_pos.csv',
                      '/home/guotai/data/phrosis_data/names/new_fold_5_pos.csv']
prediction_file = 'experiments/exp1/vgg_old_new_fold_5_pos.csv'

grnd = []
for csv_file in ground_truth_files:
    temp_grnd = np.genfromtxt(csv_file, delimiter=',')
    temp_grnd = temp_grnd[1:, 1:]
    grnd.append(temp_grnd)
grnd = np.concatenate(grnd, axis = 0)
print('ground truth shape', grnd.shape)

pred = np.genfromtxt(prediction_file, delimiter=',')
pred = pred[1:, 1:]

N = grnd.shape[0]
grnd_sum = np.sum(grnd, axis = 0)
baseline = (N - grnd_sum)/N

# class level accuracy 
correct = pred == grnd
correct_sum = np.sum(correct, axis = 0)
accuracy = correct_sum/N

#  class level sensitivity
pos = np.sum(pred, axis = 0) + 1e-5
trp = np.sum(pred * grnd, axis = 0) + 1e-5
sensitivity = trp/pos

# class level specificity
neg = np.sum(1 - pred, axis = 0) + 1e-5
trn = np.sum((1-pred)*(1-grnd), axis = 0) + 1e-5
specificity = trn/neg

# image level accuracy
correct_sum = np.sum(correct, axis = 1) == grnd.shape[1]
img_acc = np.sum(correct_sum)/N

np.set_printoptions(precision = 3)
print('positive num ', grnd_sum)
print('baseline acc ', baseline)
print('cnn acc      ', accuracy)
print('cnn sensiti  ', sensitivity)
print('cnn specifi  ', specificity)
print('image level acc', img_acc)
