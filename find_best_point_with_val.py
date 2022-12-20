import argparse
import numpy as np
import pandas as pd
import math


def evaluate(array, beta=3):
	#Compute average accuracy, i.e. accuracy of the model on the whole test set

	a = np.average(array)/100

	#Compute evenness score
	#Compute entropy

	h = 0
	for entry in array:
	 h+=(entry/np.sum(array))*np.log(entry/np.sum(array))
	 
	h *= -1

	#Compute hmax

	hmax = np.log(len(array))

	#J

	j = h/hmax

	hmean = (1+beta**2)*a*j/(beta**2*a+j)
	
	return hmean, a, j


parser = argparse.ArgumentParser(description='Looking for best point')
parser.add_argument('--dir', type=str, default='/tmp/eval', metavar='DIR',
                    help='training directory (default: /tmp/eval)')
parser.add_argument('--subject', type=int, default=25, 
                    help='subject to test (default: 25)')
parser.add_argument('--beta', type=float, default=3, 
                    help='subject to test (default: 25)')
parser.add_argument('--dataset', type=str, default='Tamil', metavar='DATASET',
                    help='dataset name (default: Tamil)')        

args = parser.parse_args()
f = args.dir + '{}_Subject_{}.csv'.format(args.dataset, args.subject)
np.set_printoptions(suppress=True)
if args.dataset == 'Tamil':
    total_set = pd.read_csv(f, sep=',', header = 0)
    test_cols = total_set.columns[-24:-12]
    val_cols = total_set.columns[-12:]
    val_set = total_set[val_cols].values
    test_set = total_set[test_cols].values
else:
    val_set = pd.read_csv(f, sep=',', header = 0, usecols = ['Val_Subj_0','Val_Subj_1', 'Val_Subj_2', 'Val_Subj_3', 'Val_Subj_4']).values
    test_set = pd.read_csv(f, sep=',', header = 0, usecols = ['Test_Subj_0','Test_Subj_1', 'Test_Subj_2', 'Test_Subj_3', 'Test_Subj_4']).values

val_acc = np.asarray(val_set)


val_scores = []
val_values = []

for entry in val_acc:
	
	hmean, a, j = evaluate(entry, args.beta)
	val_scores.append(hmean)
	val_values.append((a,j))

print("-"*20)

max_val_idx = val_scores.index(max(val_scores))
val_point = val_acc[max_val_idx]
print('validation point : {}'.format(val_point))
print('validation idx : {}'.format(max_val_idx))
print('hmean: {0:1.5f}; a : {1:1.4f}; j: {2:1.5f}'.format(val_scores[max_val_idx], val_values[max_val_idx][0],val_values[max_val_idx][1]))

te_acc = np.asarray(test_set)


te_scores = []
te_values = []

for entry in te_acc:
	
	hmean, a, j = evaluate(entry, args.beta)
	te_scores.append(hmean)
	te_values.append((a,j))


max_idx = te_scores.index(max(te_scores))
point = te_acc[max_idx]
print('best test point : {}'.format(point))
print('best test idx : {}'.format(max_idx))
print('hmean: {0:1.5f}; a : {1:1.4f}; j: {2:1.5f}'.format(te_scores[max_idx], te_values[max_idx][0],te_values[max_idx][1]))

print("-"*20)
print("Test results")
print("-"*20)
point_val = te_acc[max_val_idx]
print('according to val point : {}'.format(point_val))
print('val idx : {}'.format(max_val_idx))
print('hmean: {0:1.5f}; a : {1:1.4f}; j: {2:1.5f}'.format(te_scores[max_val_idx], te_values[max_val_idx][0],te_values[max_val_idx][1]))

print("-"*20)

print('point 0 = {}'.format(te_acc[0]))
print('hmean: {0:1.5f}; a : {1:1.4f}; j: {2:1.5f}'.format(te_scores[0], te_values[0][0],te_values[0][1]))
print('---'*20)
print('last point : {}'.format(te_acc[-1]))
print('hmean: {0:1.5f}; a : {1:1.4f}; j: {2:1.5f}'.format(te_scores[-1], te_values[-1][0],te_values[-1][1]))
print('---'*20)


