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
parser.add_argument('--beta', type=int, default=3, 
                    help='subject to test (default: 25)')
parser.add_argument('--dataset', type=str, default='Tamil', metavar='DATASET',
                    help='dataset name (default: Tamil)')        

args = parser.parse_args()
f = args.dir + '{}_Subject_{}.csv'.format(args.dataset, args.subject)
total_set = pd.read_csv(f, sep=',', header = 0)
if args.dataset == 'Tamil':
    cols= total_set.columns[-24:].values
else:
    cols = total_set.columns[-5:].values
acc = np.asarray(total_set[cols].values)


scores = []
values = []

for entry in acc:
	
	hmean, a, j = evaluate(entry, args.beta)
	scores.append(hmean)
	values.append((a,j))


print('point 0 = {}'.format(acc[0]))
print('hmean: {0:1.5f}; a : {1:1.4f}; j: {2:1.5f}'.format(scores[0], values[0][0],values[0][1]))
print('---'*20)
print('last point : {}'.format(acc[-1]))
print('hmean: {0:1.5f}; a : {1:1.4f}; j: {2:1.5f}'.format(scores[-1], values[-1][0],values[-1][1]))
print('---'*20)

max_idx = scores.index(max(scores))
point = acc[max_idx]
print('point : {}'.format(point))
print('idx : {}'.format(max_idx))
print('hmean: {0:1.5f}; a : {1:1.4f}; j: {2:1.5f}'.format(scores[max_idx], values[max_idx][0],values[max_idx][1]))


