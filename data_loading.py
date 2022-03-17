"""
data.py
Loads data and builds a dataset object given
the id of the subject whose data should be the test set.
"""

import os
import sys
import tarfile
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections import defaultdict
import webdataset as wds
from torchvision import transforms

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from math import floor, inf

data_directory = Path(__file__).parent/"AAAIActRec/data_loading/npy"
tamil_directory = Path(__file__).parent/"wds_tamil/tars"

_UCI_DATA_FILES = {
    "X": data_directory/"UCI_full_data_tensor.npy",
    "y": data_directory/"UCI_full_act_labels.npy",
    "s": data_directory/"UCI_full_subj_labels.npy"
}

_UTK_DATA_FILES = {
    "X": data_directory/"UTK_X.npy",
    "y": data_directory/"UTK_y.npy",
    "s": data_directory/"UTK_subj.npy"
}

_UCI_labels = [
        "Walking",
        "WalkingUpstairs",
        "WalkingDownstairs",
        "Sitting",
        "Standing",
        "Laying"
    ]

_UTK_labels = [
        "Basketball",
        "ComputerWork",
        "Run",
        "StairWalking",
        "SupineRest",
        "Sweeping",
        "TableCleaning",
        "Tennis",
        "Walking"
    ]

_TAMIL_labels = range(156)

def ActivityRecognitionDataset(
    dataset="UTK", 
    cross_val_subject_id_start=25,  
    scale=True,
    batch_size = 125
):
    """
    Loads numpy data, then separates out training and test data and 
    creates torch DataLoader dataset objects for model training and evaluation.
    
    Inputs
    -------
    cross_val_subject_id: integer. For current dataset, 
        should be between 0 and 29.
    batch_size : integer. 
    Returns
    -------
    train_dataset: torch.utils.data.DataLoader consisting of all training 
        data (not from subject cross_val_subj_id)
    test_dataset: array of torch.utils.data.DataLoader-s consisting of all evaluation
        data (from subject cross_val_subj_id)
    """

    if dataset == "UCI":
        data_files = _UCI_DATA_FILES
        label_names = _UCI_labels
    else:
        data_files = _UTK_DATA_FILES
        label_names = _UTK_labels

    X = np.load(data_files["X"], allow_pickle=True)
    y = np.load(data_files["y"], allow_pickle=True)
    ids = np.load(data_files["s"], allow_pickle=True)
    #np.savetxt("foo.csv", np.asarray(ids), delimiter=",")
    int_labels = y.astype(np.int64)
    # Numpy sorts the unique items by default
    unique_ids = np.unique(ids)
    unique_labels = np.unique(int_labels)
    np.random.seed(100)

    # Get one-hot labels tensor
    one_hot_labels = F.one_hot(torch.from_numpy(int_labels), len(unique_labels))

    # The ids in the numpy array are integers from 101 to 130.
    # We remap them to integers from 0 to 29 here so that
    # the calling function doesn't need to know what's
    # in the ids array when this function is called.
    test_mask = (ids >= unique_ids[cross_val_subject_id_start]) & (ids <= unique_ids[cross_val_subject_id_start+4])
    
    
    # In leave-one-out cross-validation, we aren't using a 
    # validation set (we should probably skim off a validation set),
    # so what's not in training is in testing.
    train_mask = np.invert(test_mask)

    # Scale the data for use in the model. Use StandardScaler to 
    # fit the training set, then transform both the training
    # and test sets with those statistics. In order to do this,
    # We reshape the data such that each feature exists in one 
    # long column (for example, the X axis acceleration for all
    # timestamps in the training set sit in one column, instead of 
    # being organized by activity bout). One the scaling is complete,
    # we reshape back.

    X_train_raw = X[train_mask]
    X_test_raw = []
    Y_test = []
    #test_mask
    for entry in range(cross_val_subject_id_start, cross_val_subject_id_start+5):
        test_mask = ids == unique_ids[entry]
        X_test_raw.append(X[test_mask])
        Y_test.append(one_hot_labels[test_mask])
        
    
    fn = lambda array, elem: np.where(array == elem)[0][0]
    
    new_indices = [ fn(unique_ids, x) for x in ids]
    loss_masks = F.one_hot(torch.as_tensor(new_indices), len(unique_ids))

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(
            X_train_raw.reshape(
                (X_train_raw.shape[0] * X_train_raw.shape[1], 
                X_train_raw.shape[2])
            )
        )
        X_train = X_train.reshape((X_train_raw.shape[0], X_train_raw.shape[2], X_train_raw.shape[1] ))
        X_test =[]
        num_elements_test=0
        num_elem_test_arr = []
        for idx, entry in enumerate(X_test_raw):
            X_test_i = scaler.transform(
                entry.reshape(
                    (entry.shape[0] * entry.shape[1], 
                    entry.shape[2])
                )
            )
            num_elements_test+= entry.shape[0]
            num_elem_test_arr.append(entry.shape[0])
            X_test.append(X_test_i.reshape((entry.shape[0], entry.shape[2], entry.shape[1])))
    else:
        X_train, X_test = X_train_raw, X_test_raw
        
    train_dataset = torch.utils.data.TensorDataset(
                                                                        torch.from_numpy(X_train), 
                                                                        one_hot_labels[train_mask] 
                                                                        )
        
    test_dataset = []
    for idx,entry in enumerate(X_test):
                test_dataset.append(
                    torch.utils.data.TensorDataset(
                        torch.from_numpy(entry), 
                        Y_test[idx]
                )
            )

    return {
               'train': torch.utils.data.DataLoader(
                   train_dataset,
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=4
               ),
               'test':  test_dataset
           }, len(label_names),  num_elements_test, num_elem_test_arr
   #total test size, size of each  test set in array


def TamilLettersDataset(
    dataset="Tamil", 
    cross_val_subject_id_start=25, 
    batch_size = 125,
    step = 24
):
    """
    Loads numpy data , then separates out training and test data and 
    creates torch DataLoader dataset objects for model training and evaluation.
    
    Inputs
    -------
    cross_val_subject_id_start: integer. For current dataset, 
        should be one of [25, 49, 73, 97, 121, 145].
    
    Returns
    -------
    train_dataset: tf.data.DataSet consisting of all training 
        data (not from subject cross_val_subj_id)
    test_dataset: tf.data.DataSet consisting of all evaluation
        data (from subject cross_val_subj_id)
    """
    def identity(x):
        return x
    ids = list(range(169))
    labels = list(range(156))
    arr  = []
    train_size = 0
    url= "usr_{}.tar" 
    # The ids in the numpy array are integers [16 to 54, 100 to 229].
    # We remap them to integers from 0 to 168 here so that
    # the calling function doesn't need to know what's
    # in the ids array when this function is called.
    for entry in ids:
        if not entry in list(range(cross_val_subject_id_start,cross_val_subject_id_start+step)):
            arr.append(entry)
            with tarfile.open(os.path.join(tamil_directory,url.format(entry))) as archive:
                train_size += sum(1 for member in archive if member.isreg())
            
    train_size //= 2
    usrs = ",".join(map(str,arr))
    
    url_train = os.path.join(tamil_directory,"usr_{"+usrs+"}.tar" )   
    #url_train = os.path.join(tamil_directory,"usr_16.tar" )   
    preproc = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_dataset = wds.Dataset(url_train).decode("torch")\
                                                            .shuffle(100)\
                                                            .to_tuple("png","cls")\
                                                            .batched(batch_size)
    

    dataloader = torch.utils.data.DataLoader(train_dataset, num_workers = 0, batch_size = None)
    test_dataset = []
    num_elements_test = 0
    num_elem_test_arr = []
    for idx in range(cross_val_subject_id_start, cross_val_subject_id_start+step):
        url =  os.path.join( tamil_directory,"usr_{}.tar".format(idx))
        size = 0
        dataset = wds.Dataset(url).decode("torch")\
                                                            .shuffle(100)\
                                                            .to_tuple("png","cls")\
                                                            .batched(batch_size)
        with tarfile.open(os.path.join(tamil_directory,url)) as archive:
                size += sum(1 for member in archive if member.isreg())
        num_elements_test += size
        num_elem_test_arr.append(size//2)
        test_dataset.append(dataset)
            
    return {
               'train': dataloader,
               'test':  test_dataset
           }, len(labels),  num_elements_test//2, num_elem_test_arr, train_size
   #number of labels, total test size, size of each  test set in array   
