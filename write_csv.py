"""
Script for create csv files required for training and testing
"""
import os
import csv
import numpy as np
import pandas as pd 
import random

def random_split_dataset_for_exp1(csv_all, trainval_csv, test_csv, n_test):
    """
    randomly split the entire dataset into n_test for testing and 
    the others for training + validation.
    """
    
    df = pd.read_csv(csv_all)
    df = df.sample(frac = 1) # random shuffle
    df_trainval = df.iloc[:-n_test]
    df_test     = df.iloc[-n_test:]
    df_trainval = df_trainval.sort_values("ID")
    df_test     = df_test.sort_values("ID")
    df_trainval.to_csv(trainval_csv, index = False)
    df_test.to_csv(test_csv, index = False)


def random_split_trainval_to_five_fold(trainval_csv):
    """ 
    for five-fold cross validation during traning
    """
    config_dir = "/".join(trainval_csv.split("/")[:-1])
    with open(trainval_csv, 'r') as f:
        lines = f.readlines()
    lines_data = lines[1:]
    for fi in range(5):
        random.shuffle(lines_data)
        n_data  = len(lines_data)
        n_train = int(n_data * 0.85)
        data_train = lines_data[:n_train]
        data_valid = lines_data[n_train:]
        data_train = sorted(data_train)
        data_valid = sorted(data_valid)
        output_dir = config_dir + "/f{0:}".format(fi+1)
        if(not os.path.isdir(output_dir)):
            os.mkdir(output_dir)
        train_csv = output_dir + "/data_train_f{0:}.csv".format(fi+1)
        with open(train_csv, 'w') as f:
            f.writelines(lines[:1] + data_train)

        valid_csv = config_dir + "/f{0:}/data_valid_f{0:}.csv".format(fi+1)
        with open(valid_csv, 'w') as f:
            f.writelines(lines[:1] + data_valid)

def extract_positve_samples(input_csv):
    """ selecting positive samples to train model 2
    """
    df = pd.read_csv(input_csv)
    pos_index = []
    for h in range(df.shape[0]):
        label = df.iloc[h, 1:]
        if(np.asarray(label).max() > 0):
            pos_index.append(h)
    df_pos  = df.iloc[pos_index, :]
    pos_csv = input_csv.replace(".csv", "_pos.csv")
    df_pos.to_csv(pos_csv, index = False)

def get_trainval_dataset_for_exp2():
    """
    In experiment 2, only use images from the new system for training/validation.
    Remove images in the old system from the trainval.csv"""
    
    trainval_csv1 = "exp1/fine_model/config/data_trainval.csv"
    trainval_csv2 = "exp2/fine_model/config/data_trainval_raw.csv"
    trainval_csv2_pos = "exp2/fine_model/config/data_trainval.csv"
    trainval_df1  = pd.read_csv(trainval_csv1)
    valid_index, valid_pos_index = [], []
    for h in range(trainval_df1.shape[0]):
        name = trainval_df1["ID"][h]
        if("data_a" in name):
            valid_index.append(h)
            label = trainval_df1.iloc[h, 1:]
            if(np.asarray(label).max() > 0):
                valid_pos_index.append(h)
    trainval_df2 = trainval_df1.iloc[valid_index, :]
    trainval_df2.to_csv(trainval_csv2, index = False)
    trainval_df2_pos = trainval_df1.iloc[valid_pos_index, :]
    trainval_df2_pos.to_csv(trainval_csv2_pos, index = False)
    config_dir = "exp2/fine_model/config"
    random_split_trainval_to_five_fold(config_dir)

def get_test_dataset_for_exp2():
    """
    Create internal testing images
    """        
    test_csv1 = "exp1/fine_model/config/data_test.csv"
    test_csv2 = "exp2/fine_model/config/data_test_internal.csv"
    test_df1  = pd.read_csv(test_csv1)
    internal_index = []
    for h in range(test_df1.shape[0]):
        name = test_df1["ID"][h]
        if("data_a" in name):
            internal_index.append(h)
    internal_df = test_df1.iloc[internal_index, :]
    internal_df.to_csv(test_csv2, index = False)

def get_maximal_label(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    id = df["ID"]
    lab = np.asarray(df.iloc[:, 1:])
    lab = lab.max(axis = 1)
    out_dict = {"image": id, "label": lab}
    out_df = pd.DataFrame.from_dict(out_dict)
    out_df.to_csv(output_csv, index = False)

def create_csvs_for_coarse_model():
    input_dir  = "./exp1/fine_model/config"
    output_dir = "./exp1/coarse_model/config"
    for stage in ["trainval", "test"]:
        input_csv = input_dir + "/data_{0:}.csv".format(stage) 
        output_csv= output_dir + "/data_{0:}.csv".format(stage) 
        get_maximal_label(input_csv, output_csv)
    

if __name__ == "__main__":
    """
    demo for data split and creating csv files for training
    """
    ## For experiment 1
    # randomly split all the data into trainval and testing
    csv_all      = "data/dataset_demo.csv"
    trainval_csv = "exp1/fine_model/config/data_trainval.csv"
    test_csv     = "exp1/fine_model/config/data_test.csv"
    n_test       = 80
    random_split_dataset_for_exp1(csv_all, trainval_csv, test_csv, n_test)
    extract_positve_samples(trainval_csv)
    # randomly split the trainval set into training and validation
    # for five-fold cross validation
    create_csvs_for_coarse_model()
    trainval_csv = "exp1/coarse_model/config/data_trainval.csv"
    random_split_trainval_to_five_fold(trainval_csv)
    trainval_csv = "exp1/fine_model/config/data_trainval_pos.csv"
    random_split_trainval_to_five_fold(trainval_csv)
    

    
    ## For experiment 2
    # get_trainval_dataset_for_exp2()
    # get_test_dataset_for_exp2()