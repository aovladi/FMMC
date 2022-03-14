import numpy as np
import os
import torch
import torch.nn.functional as F

import curves


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None, var_training=False, alpha = 0.0, one_hot = True, size_wds = 0, num_iters = 0):
    loss_sum = 0.0
    correct = 0.0
    if size_wds == 0:
        num_iters = len(train_loader)
    else:
        num_iters = num_iters
    #print(train_loader)
    model=model.cuda().float()
    model.train()
    for it, (inpt, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(it / num_iters)
            adjust_learning_rate(optimizer, lr)
        #print(inpt.shape)
        inpt = inpt.cuda()
        target = target.cuda()
        if one_hot:
            target = torch.max(target, 1)[1]
        output = model(inpt.float())
        loss = criterion(output, target, reduction='none') if var_training == True else criterion(output, target) 
        size = loss.size()
        if len(size)>0:
            mean = torch.mean(loss)
            greater = torch.gt(abs(loss), alpha*mean).int()    
            greater += 1
            loss = torch.mean(loss*greater)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * inpt.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss':  loss_sum / len(train_loader.dataset) if size_wds==0 else loss_sum / size_wds,
        'accuracy': correct * 100.0 / len(train_loader.dataset) if size_wds==0 else correct * 100.0 / size_wds,
    }


def test(test_loader, model, criterion, regularizer=None, one_hot = True, size_wds = 0,**kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0
    model=model.float()
    model.eval()

    for input, target in test_loader:
        input = input.cuda()
        target = target.cuda()
        if one_hot:
            target = torch.max(target, 1)[1]
        output = model(input.float(), **kwargs)
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset) if size_wds==0 else nll_sum / size_wds,
        'loss': loss_sum / len(test_loader.dataset) if size_wds==0 else loss_sum / size_wds,
        'accuracy': correct * 100.0 / len(test_loader.dataset) if size_wds==0 else correct * 100.0 / size_wds
    }

def multi_test(test_loader, model, criterion, num_elements_test,num_elem_test_arr, regularizer=None,\
                        batch_size = 125, one_hot = True, var_training = False, alpha = 0.0, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0
    correct_arr = []
    
    model = model.float()
    model.eval()
    prev_id = -1
    for idx, entry in enumerate(test_loader):
        ds = torch.utils.data.DataLoader(
                   entry,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=0
               )
        for input, target in ds:
            input = input.cuda()
            target = target.cuda()
            if one_hot:
                target = torch.max(target, 1)[1]
            
            output = model(input.float(), **kwargs)
            nll = criterion(output, target)
            size = nll.size()
            if len(size)>0:
                nll = torch.mean(nll)
            loss = nll.clone()
            if regularizer is not None:
                loss += regularizer(model)

            nll_sum += nll.item() * input.size(0)
            loss_sum += loss.item() * input.size(0)
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            if prev_id != idx:
                if prev_id >= 0 :
                    correct_arr.append(correct_cur *100 / num_elem_test_arr[prev_id])
                correct_cur = 0
                prev_id  = idx
            correct_cur += pred.eq(target.data.view_as(pred)).sum().item()
           
    correct_arr.append(correct_cur *100 / num_elem_test_arr[-1])      
    return {
        'nll': nll_sum / num_elements_test,
        'loss': loss_sum / num_elements_test,
        'accuracy': correct * 100.0 / num_elements_test,
        'std' : np.std(correct_arr, ddof = 1) , # sample std
        'test_arr':correct_arr
    }


def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda()
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cuda().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
        
    model = model.float()
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda()
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input.float(), **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))
