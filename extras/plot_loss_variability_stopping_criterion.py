'''
************************************************************************************
* This is a code which plots the whisker plot of the variability across all epochs
* Variability across all epochs is important to understand the stopping criterion 
* 
* Inputs required are as below
* folder_path(-p)     | path of the folder containing the log_loss_*.txt text files
* shuffle(sh)         | flag that specifies whether shuffle of datasets was used
*                  during training of model
* loss_func(-l)       | specifies the loss function used during training of the model.
* train_batch_size(-trb)|specifies the loss function used during training of the model.
**************************************************************************************
'''
import numpy as np
import os, fnmatch
import pandas   as pd
import matplotlib.pyplot as plt
import argparse as argp
from IPython.display import display

# list of the loss functions used in training of the Vnet model
list_loss_func   = ['Sorensen_Dice','WtSorensen_Dice']


def get_dataframe(file):
    #define an empty list to collect the training loss in each epoch files
    loss= []
    with open(file, "r") as infile:
        for line in infile:
            #print(line)
            loss.append((line.split()[1]))
    loss.pop(0)
    #print(loss)
    pd_loss_list = pd.Series(loss)
    return pd_loss_list


def get_df_epochs(path):
    
     
    file_list = fnmatch.filter(sorted(os.listdir(path)), '*train.txt')
    folder_name = os.path.basename(path)
    #print("file_list",file_list)
    csv_flname = 'epoch_loss_' +folder_name
    df_epoch_loss = pd.DataFrame()
    idx = 0
    for file in file_list:
        path_str   = os.path.join(path, file)
        #print(path_str)
        pd_loss_list = get_dataframe(path_str)
        df_epoch_loss[idx] = pd_loss_list.values
        idx =idx+1
    #print(df_epoch_loss)
    df_epoch_loss.to_csv(csv_flname + '.csv')

    return csv_flname 

def get_args():

    parser = argp.ArgumentParser(prog = 'plot_loss_variability_stopping_criterion.py',
                                    formatter_class=argp.RawTextHelpFormatter)

    # folder_path is the path of the folder containing the log_loss_*.txt text files
    parser.add_argument("-p", "--folder_path")
    # shuffle is a flag that specifies whether shuffle of datasets was used
    # during training of model
    parser.add_argument("-sh", "--shuffle")
    # loss_func specifies the loss function used during training of the model.
    parser.add_argument("-l", "--loss_func")
    # train_batch_size  specifies the loss function used during training of the model.
    parser.add_argument("-trb","--train_batch_size")

    return parser.parse_args()

def plot_loss_variability(csv_flname,folder_name,shuffle,loss_func,train_batch_size):
    fig = plt.figure(figsize=(10, 10))
    df  = pd.read_csv (csv_flname +'.csv')
    # to avoid :: pandas DataFrame "no numeric data to plot" error
    df  = df.astype(float)
    boxplot = df.boxplot(column=['10','20','30','40','50','60','70','80','90','100']
                                    , rot=45, fontsize=17)
    plt.grid(True)
    plt.ylabel("Loss",fontsize=17)
    plt.title("loss_over_epochs\n \n  \
        folder_name = {folder_name} || loss_func = {loss_func} || batch_size = {train_batch_size} || shuffle = {shuffle}". 
                            format(folder_name = folder_name, loss_func=loss_func, \
                             train_batch_size=train_batch_size, shuffle= shuffle))
    
    fig.savefig(folder_name+ '.jpg', format="jpg")
    plt.show()


def main():

    args        = get_args()
    # folder_path is the path of the folder containing the log_loss_*.txt text files
    folder_path = args.folder_path
    # folder name containing the log_loss_*.txt text files
    folder_name = os.path.basename(folder_path)
    
    # shuffle is a flag that specifies whether shuffle of datasets was used
    # during training of model
    shuffle     = args.shuffle 
    # loss_func specifies the loss function used during training of the model.
    loss_func   = args.loss_func 

    train_batch_size = args.train_batch_size 
    #delete
    
    print('1) folder_path =',folder_path)
    print('2) folder_name =',folder_name)
    print('3) shuffle =',shuffle)
    print('4) loss_func =',loss_func)
    print('5) train_batch_size =',train_batch_size)

    
    csv_flname = get_df_epochs(folder_path)
    #display(df_epoch_loss.head())
    plot_loss_variability(csv_flname, folder_name, shuffle, loss_func, train_batch_size)

  

main()