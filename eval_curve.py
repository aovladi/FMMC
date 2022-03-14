import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F


import data_loading
import models
import curves
import utils
import datetime
import csv

time = datetime.datetime.now()

parser = argparse.ArgumentParser(description='DNN curve evaluation')
parser.add_argument('--dir', type=str, default='/tmp/eval', metavar='DIR',
                    help='training directory (default: /tmp/eval)')

parser.add_argument('--num_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--subject', type=int, default=25, 
                    help='subject to test (default: 25)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True


one_hot = True
if args.dataset == "ActRec": 
    loaders, num_classes, num_elements_test , num_elem_test_arr= data_loading.ActivityRecognitionDataset(\
                                                                                                                        batch_size=args.batch_size,\
                                                                                                                        cross_val_subject_id_start= args.subject)
else:
    loaders, num_classes, num_elements_test , num_elem_test_arr, train_size= data_loading.TamilLettersDataset(\
                                                                                                                        batch_size=args.batch_size,\
                                                                                                                        cross_val_subject_id_start= args.subject)
    one_hot = False

architecture = getattr(models, args.model)
curve = getattr(curves, args.curve)
model = curves.CurveNet(
    num_classes,
    curve,
    architecture.curve,
    args.num_bends,
    architecture_kwargs=architecture.kwargs,
)
model.cpu()
checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['model_state'], strict=False)

criterion = F.cross_entropy
regularizer = curves.l2_regularizer(args.wd)

T = args.num_points
ts = np.linspace(0.0, 1.0, T)
tr_loss = np.zeros(T)
tr_nll = np.zeros(T)
tr_acc = np.zeros(T)
te_loss = np.zeros(T)
te_nll = np.zeros(T)
te_acc = np.zeros(T)
tr_err = np.zeros(T)
te_err = np.zeros(T)
te_std = np.zeros(T)
dl = np.zeros(T)

previous_weights = None

columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)', 'Test std','Subj_0','Subj_1', 'Subj_2', 'Subj_3', 'Subj_4']
if  args.dataset == "Tamil": 
    for idx in range(5, 24):
        columns.append('Subj_{}'.format(idx))
# name of csv file  
filename = args.dir + "{}_Subject_{}_{}.csv".format(args.dataset, args.subject, time.strftime("%c"))
    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
    # writing the fields  
    csvwriter.writerow(columns)  
    
t = torch.FloatTensor([0.0]).cpu()
for i, t_value in enumerate(ts):
    t.data.fill_(t_value)
    weights = model.weights(t)
    if previous_weights is not None:
        dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
    previous_weights = weights.copy()
    utils.update_bn(loaders['train'], model, t=t)
    if args.dataset == "ActRec": 
        tr_res = utils.test(loaders['train'], model, criterion, regularizer,  t=t)
    else:
        tr_res = utils.test(loaders['train'], model, criterion, regularizer, one_hot = one_hot, size_wds = train_size, t=t)
    
    if args.dataset == "Tamil":
            args.batch_size = None
    te_res = utils.multi_test(loaders['test'], model, criterion, num_elements_test, num_elem_test_arr, regularizer,\
                                                 batch_size=args.batch_size, one_hot = one_hot,t=t)
    tr_loss[i] = tr_res['loss']
    tr_nll[i] = tr_res['nll']
    tr_acc[i] = tr_res['accuracy']
    tr_err[i] = 100.0 - tr_acc[i]
    te_loss[i] = te_res['loss']
    te_nll[i] = te_res['nll']
    te_acc[i] = te_res['accuracy']
    te_err[i] = 100.0 - te_acc[i]
    te_std[i] = te_res['std']

    values = [t, tr_loss[i], tr_nll[i], tr_err[i], te_nll[i], te_err[i], te_std[i] ] #,te_res['test_arr'][0],te_res['test_arr'][1],\
    #te_res['test_arr'][2],te_res['test_arr'][3],te_res['test_arr'][4]]
    for entry in te_res['test_arr']:
        #print(entry)
        values.append(entry)
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
    if i % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
    
    # writing to csv file  
    with open(filename, 'a') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the fields  
        csvwriter.writerow([float(i) for i in values])  
        
    torch.save(model.state_dict(), args.dir+'model'+str(i)+'.pt')
   
def stats(values, dl):
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
    return min, max, avg, int


tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = stats(tr_loss, dl)
tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = stats(tr_nll, dl)
tr_err_min, tr_err_max, tr_err_avg, tr_err_int = stats(tr_err, dl)

te_loss_min, te_loss_max, te_loss_avg, te_loss_int = stats(te_loss, dl)
te_nll_min, te_nll_max, te_nll_avg, te_nll_int = stats(te_nll, dl)
te_err_min, te_err_max, te_err_avg, te_err_int = stats(te_err, dl)
te_std_min, te_std_max, te_std_avg, te_std_int = stats(te_std, dl)

print('Length: %.2f' % np.sum(dl))
print(tabulate.tabulate([
        ['train loss', tr_loss[0], tr_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
        ['train error (%)', tr_err[0], tr_err[-1], tr_err_min, tr_err_max, tr_err_avg, tr_err_int],
        ['test nll', te_nll[0], te_nll[-1], te_nll_min, te_nll_max, te_nll_avg, te_nll_int],
        ['test error (%)', te_err[0], te_err[-1], te_err_min, te_err_max, te_err_avg, te_err_int],
        ['test std', te_std[0], te_std[-1], te_std_min, te_std_max, te_std_avg, te_std_int]
    ], [
        '', 'start', 'end', 'min', 'max', 'avg', 'int'
    ], tablefmt='simple', floatfmt='10.4f'))

np.savez(
    os.path.join(args.dir, 'curve.npz'),
    ts=ts,
    dl=dl,
    tr_loss=tr_loss,
    tr_loss_min=tr_loss_min,
    tr_loss_max=tr_loss_max,
    tr_loss_avg=tr_loss_avg,
    tr_loss_int=tr_loss_int,
    tr_nll=tr_nll,
    tr_nll_min=tr_nll_min,
    tr_nll_max=tr_nll_max,
    tr_nll_avg=tr_nll_avg,
    tr_nll_int=tr_nll_int,
    tr_acc=tr_acc,
    tr_err=tr_err,
    tr_err_min=tr_err_min,
    tr_err_max=tr_err_max,
    tr_err_avg=tr_err_avg,
    tr_err_int=tr_err_int,
    te_loss=te_loss,
    te_loss_min=te_loss_min,
    te_loss_max=te_loss_max,
    te_loss_avg=te_loss_avg,
    te_loss_int=te_loss_int,
    te_nll=te_nll,
    te_nll_min=te_nll_min,
    te_nll_max=te_nll_max,
    te_nll_avg=te_nll_avg,
    te_nll_int=te_nll_int,
    te_acc=te_acc,
    te_err=te_err,
    te_err_min=te_err_min,
    te_err_max=te_err_max,
    te_err_avg=te_err_avg,
    te_err_int=te_err_int,
    te_std=te_std,
    te_std_min=te_std_min,
    te_std_max=te_std_max,
    te_std_avg=te_std_avg,
    te_std_int=te_std_int,
)
