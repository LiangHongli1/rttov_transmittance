# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 08:52:47 2018
建立rnn和lstm模型，用于透过率计算，此次试验用EC83廓线。
每个时间点输入一个气压层，共100层，因此RNN一层设置100个时间步.60条训练、23条验证
@author: ccsky
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from pathlib import Path
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
import argparse
import os
import h5py

class trans_GRU_reg():
    def __init__(self,is_training,batch_size,num_steps,input_size,output_size,\
                 hidden_units,num_layers,learning_rate,dropout_ratio,max_grad_norm):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        keep_prob = 1-dropout_ratio
        self.max_grad_norm = max_grad_norm
        #输入层，大小为[batch_size,num_steps,input_size]
        self.inputs = tf.placeholder(tf.float32,[batch_size,num_steps,input_size])
        #预期输出
        self.targets = tf.placeholder(tf.float32,[batch_size,num_steps,output_size])
        targets = tf.reshape(self.targets,[-1,output_size])
        #定义使用的循环体结构GRU，使用dropout的深层循环神经网络
        cells = [tf.nn.rnn_cell.GRUCell(num_units=n) for n in hidden_units]
        if is_training:
            gru_cell2 = tf.nn.rnn_cell.DropoutWrapper(cells[-1],output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells,state_is_tuple=True)
        #cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layers) #这里第一层和第二层隐藏结点数相同
        #初始化最初状态，全零
        self.initial_state = cell.zero_state(batch_size,tf.float32)
        
        if is_training:
            inputs = tf.nn.dropout(self.inputs,keep_prob)
        else:
            inputs = self.inputs
        
        state = self.initial_state
        outputs,state = tf.nn.dynamic_rnn(cell,inputs,initial_state=state,dtype=tf.float32)
##        outputs = []
##        state = self.initial_state
##        print('cell.state_size',cell.state_size)
##        with tf.variable_scope('RNN',reuse=tf.AUTO_REUSE):
##            for time_step in range(num_steps):
##                if time_step>0:
##                    tf.get_variable_scope().reuse_variables()
##                    #从输入数据中获取当前时刻的输入并传入LSTM结构
##                #print(inputs[:,time_step,:])
##                (cell_output,state) = cell(inputs[:,time_step,:],state)
##                outputs.append(cell_output)

        #print(outputs)      
        output = tf.reshape(outputs,[-1,hidden_units[-1]])
        weights = tf.get_variable('weight',[hidden_units[-1],output_size])
        bias = tf.get_variable('bias',[output_size])
        logits = tf.matmul(output,weights)+bias
        #print(logits)
        loss = tf.losses.mean_squared_error(targets,logits)
        #print(loss)
        self.predictions = logits #shape=[batch_size*num_steps,output_size]
        #计算得到每个batch的平均损失
        self.cost = tf.reduce_sum(loss)/batch_size
        self.final_state = state
        #只在训练时定义反向传播操作
        if not is_training: return
        trainable_variables = tf.trainable_variables()
        #print(trainable_variables)
        #通过clip_by_global_norm函数控制梯度的大小，避免梯度膨胀的问题
        grads,_ = tf.clip_by_global_norm(tf.gradients(self.cost,trainable_variables),max_grad_norm)
        #定义优化方法
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        #定义训练步骤
        self.train_op = optimizer.apply_gradients(zip(grads,trainable_variables))

        
def run_epoch(sess,model,x,y,train_op,batch_size):
    total_loss = 0.0
    #iters = 0
    num_batches = x.shape[0]//batch_size
    predicts = []
    state = sess.run(model.initial_state)
    for step in range(num_batches):
        batch_x,batch_y = x[step*batch_size:step*batch_size+batch_size],y[step*batch_size:step*batch_size+batch_size]
        loss,state,epredicts,_ = sess.run([model.cost,model.final_state,model.predictions,train_op],
                                {model.inputs:batch_x,model.targets:batch_y,model.initial_state:state})
        total_loss += loss
        #iters += model.num_steps
        predicts.append(epredicts)
##        if step%1==0:
##            print("After %d steps, loss is %.3f" % (step,loss))#把一条廓线看作一个样本则样本太少，每一个step都显示

    predicts = np.array(predicts) 
    return total_loss,predicts

def plot_loss(epoch,losses,ch,figpath):
    x = np.arange(1,epoch+1)
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams.update({'font.size':10})
    plt.plot(x,losses,'k-',linewidth=1.3)
    plt.xlabel('epoch',fontsize=12)
    plt.ylabel('train_loss',fontsize=12)
    plt.title('GRU2l_model',fontsize=12)
    figname = 'GRU2l_loss_curve_ch'+str(ch)+'.png'
    plt.savefig(figpath/figname,dpi=300)
    plt.show()

def plot_preds(trainy,train_preds,testy,test_preds,ch,figpath):
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams.update({'font.size':10})
    plt.figure(1)
    plt.scatter(trainy,train_preds,s=1.5,c='b')
    plt.plot(trainy,trainy,'r-',linewidth=1.5)
    plt.xlabel('train_trans_true',fontsize=12)
    plt.ylabel('train_trans_predictions',fontsize=12)
    plt.title('GRU2l_train',fontsize=12)
    trainfig = 'GRU2l_train_ch'+str(ch)+'.png'
    plt.savefig(figpath/trainfig,dpi=300)
    plt.show()
    
    plt.figure(2)
    plt.scatter(testy,test_preds,s=1.5,c='b')
    plt.plot(testy,testy,'r-',linewidth=1.5)
    plt.xlabel('test_trans_true',fontsize=12)
    plt.ylabel('test_trans_predictions',fontsize=12)
    plt.title('GRU2l_test',fontsize=12)
    testfig = 'GRU2l_test_ch'+str(ch)+'.png'
    plt.savefig(figpath/testfig,dpi=300)
    plt.show()

def loadata(fname):
    with h5py.File(fname,'r') as f:
        X = f['dependent/X'][:,:303]
        Y = f['dependent/Y'][:,:,1:-1]
        tao_rttov = f['dependent/transmission_RTTOV'][:,:,:-1]
        bt_rttov = f['dependent/BT_RTTOV'][:]
        bt_true = f['dependent/BT_true'][:]
        emiss = f['dependent/emissivity'][:]
        new_preds = f['dependent/predictor_RTTOV'][:]

    xtemp = np.concatenate((X.reshape((-1,3,101))[:,:,1:],new_preds),axis=1)
    # xtemp = X.reshape((-1,3,101))[:,:,1:]
    x = np.transpose(xtemp,[0,2,1]) #reshape为[n_samples,num_steps,input_size]
    y = np.transpose(Y,[1,2,0])[:,:,0][:,:,np.newaxis]#一个output为一个气压层单个通道的透过率
    random.seed(2020)
    n = x.shape[0]
    testnum = int(n*0.2)
    testindex = random.sample(range(n),testnum)
    traindex = [x for x in range(n) if x not in testindex]
    trainx = x[traindex]
    trainy = y[traindex]
    testx = x[testindex]
    testy = y[testindex]
    #preprocessing
    # minmax_scaler = preprocessing.MinMaxScaler()
    # trainx = minmax_scaler.fit_transform(trainx.reshape([-1,FLAGS.input_size*FLAGS.num_steps])).reshape([-1,FLAGS.num_steps,FLAGS.input_size])
    # testx = minmax_scaler.fit_transform(testx.reshape([-1,FLAGS.input_size*FLAGS.num_steps])).reshape([-1,FLAGS.num_steps,FLAGS.input_size])
    #print(trainx[:20])
    #print(testx[:20])
    return trainx,trainy,testx,testy
def fsave(test,preds,savepath,ch):
    fname = 'GRU2l_test_preds_ch'+str(ch)+'.xlsx'
    writer = pd.ExcelWriter(savepath/fname)
    test = pd.DataFrame(data=test,columns=['mg','wv','o3'])
    preds = pd.DataFrame(data=preds,columns=['mg','wv','o3'])
    test.to_excel(writer,sheet_name='true_ch'+str(ch))
    preds.to_excel(writer,sheet_name='preds_ch'+str(ch))
    return

def main():
    fx = Path(FLAGS.data_path)/'dataset_101L_1500.HDF'
    figpath = Path(r'G:\DL_transmitance\figures\101L\reg-results\1500profile\GRU')
    # savepath = Path(r'H:\DL_transmitance\test_results_trans')
    trainx,trainy,testx,testy = loadata(fx)
    ch = 1 #channel for regression
    initializer = tf.glorot_normal_initializer()
    with tf.variable_scope('GRU_model',reuse=None,initializer=initializer):
        train_model = trans_GRU_reg(is_training=True,
                                       batch_size=FLAGS.train_batchsize,num_steps=FLAGS.num_steps,
                                       input_size=FLAGS.input_size,output_size=FLAGS.output_size,
                                       hidden_units=FLAGS.hidden_units,num_layers=FLAGS.num_layers,
                                       learning_rate=FLAGS.learning_rate,dropout_ratio=FLAGS.dropout,max_grad_norm=FLAGS.max_grad_norm)
    with tf.variable_scope('GRU_model',reuse=True,initializer=initializer):
        eval_model = trans_GRU_reg(is_training=False,
                                       batch_size=FLAGS.test_batchsize,num_steps=FLAGS.num_steps,
                                       input_size=FLAGS.input_size,output_size=FLAGS.output_size,
                                       hidden_units=FLAGS.hidden_units,num_layers=FLAGS.num_layers,
                                       learning_rate=FLAGS.learning_rate,dropout_ratio=FLAGS.dropout,max_grad_norm=FLAGS.max_grad_norm)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        trainloss = [] #其元素为模型在每个样本上的平均损失
        for i in range(1,FLAGS.epoch+1):
            eloss,train_preds = run_epoch(sess,train_model,trainx,trainy,train_model.train_op,FLAGS.train_batchsize)
            trainloss.append(eloss)
            if i%100==0:
                print('In iteration:%d, train loss of each profile is: %.4f' % (i,eloss))
        stime = time.perf_counter()
        testloss,test_preds = run_epoch(sess,eval_model,testx,testy,tf.no_op(),FLAGS.test_batchsize)
        etime = time.perf_counter()
        testime = etime-stime
        print('the test process costs %.4f seconds.' % (testime))

    plot_loss(FLAGS.epoch,trainloss,ch,figpath)

    trainy = trainy.reshape([-1,3]).reshape(-1)
    train_preds = train_preds.reshape(-1)
    #print('trainy[:200]:\n',trainy[:200])
    #print('train_preds[:200]:\n',train_preds[:200])
    testy = testy.reshape([-1,3])
    test_preds = test_preds.reshape([-1,3])
    # fsave(testy,test_preds,savepath,ch) #保存模型在测试集上的结果，以便和用RTTOV计算的结果进行比较
    testy = testy.reshape(-1)
    test_preds = test_preds.reshape(-1)
    plot_preds(trainy,train_preds,testy,test_preds,ch,figpath)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',type=str,
                        default=r'G:\DL_transmitance\revised datasets',
                        help='Transmittance data path.')
    parser.add_argument('--hidden-units',type=int,
                        default=[256,128],
                        help='Number of nodes in hidden layers.')
    parser.add_argument('--num-layers',type=int,
                        default=2,
                        help='Number of layers of GRU.')
    parser.add_argument('--num-steps',type=int,
                        default=100,
                        help='Number of steps(cells) of each layer.')#时间步长，设为大气分层的层数
    parser.add_argument('--train-batchsize',type=int,default=200)
    parser.add_argument('--test-batchsize',type=int,default=300)
    parser.add_argument('--input-size',type=int,
                        default=30,
                        help='Number of input size.')#一个样本的属性数量
    parser.add_argument('--output-size',type=int,default=1)
    parser.add_argument('--learning-rate',type=float,
                        default=0.1,help='Learning rates when optimizing.')
    parser.add_argument("--epoch",type=int,default=1500,help='Epoch for training.')
    parser.add_argument("--dropout", type=float,
                        default=0.,
                        help="Drop out ratio.") #当前数据量不够，可能欠拟合，所以暂时不用dropout
    parser.add_argument("--clip",type=float,
                        default=0.25,
                        help="Gradient clipping ratio.")
    parser.add_argument("--max-grad-norm",type=int,default=5,
                        help="Max gradient to be clipped.")
    parser.add_argument(
      "--no-use-cudnn-rnn",
      action="store_true",
      default=False,
      help="Disable the fast CuDNN RNN (when no gpu)")
    
    FLAGS, unparsed = parser.parse_known_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
