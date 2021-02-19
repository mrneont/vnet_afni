#import numpy 
import lib_ml_train as mlt
 

if __name__ == '__main__':

    data_path = '/Users/yamunasn/Vnet_afni/dataset/pretrain_vnet_res8'
    epochs=5
    lr=0.001
    mlt.train_net(data_path,epochs,lr)
