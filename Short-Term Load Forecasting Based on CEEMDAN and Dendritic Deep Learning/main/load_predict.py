import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import datetime
import warnings
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib as mpl
from tensorflow.keras.layers import Dense, LSTM, GRU, Layer, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PyEMD import CEEMDAN, Visualisation
import scipy
import statsmodels as sm
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from math import sqrt
warnings.filterwarnings('ignore')
from pandas import concat, DataFrame
from keras.optimizers import adam_v2


class DNMLayer(Layer):
    def __init__(self, size_out, M, synapse_activation=tf.nn.sigmoid, activation=None, **kwargs):
        super(DNMLayer, self).__init__(**kwargs)
        self.size_out = size_out
        self.M = M
        self.synapse_activation = synapse_activation
        self.activation = activation
    def build(self, input_shape):
        size_in = input_shape[-1]
        self.W = self.add_weight(name="W", shape=(size_in, self.size_out * self.M),
                                 initializer=tf.keras.initializers.TruncatedNormal(stddev=tf.sqrt(2 / (self.size_out + size_in))),
                                 trainable=True) 
        self.b = self.add_weight(name="b", shape=(self.size_out * self.M,),
                                 initializer=tf.keras.initializers.Zeros(),
                                 trainable=True)
        self.DNM_weight = self.add_weight(name="DNM_weight", shape=(self.size_out, self.M),
                                          initializer=tf.keras.initializers.TruncatedNormal(stddev=tf.sqrt(2 / (self.size_out + self.M))),
                                          trainable=True)
        self.k = self.add_weight(name="k", shape=(), initializer=tf.constant_initializer(0.1), trainable=True)
        # self.k = self.add_weight(name="k", shape=(), initializer=tf.keras.initializers.TruncatedNormal(mean = 0.5, stddev = 0.1), trainable=True)
        super(DNMLayer, self).build(input_shape) 

    def call(self, inputs):
        wx_plus_b = tf.multiply(tf.add(tf.matmul(inputs, self.W), self.b), self.k)
        wx_plus_b = self.synapse_activation(wx_plus_b)

        wx_plus_b = tf.reshape(wx_plus_b, [-1, self.size_out, self.M])
        DNM_fc = tf.multiply(wx_plus_b, self.DNM_weight)
        out = tf.reduce_sum(DNM_fc, axis=2)

        if self.activation:
            out = self.activation(out)
        return out
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size_out': self.size_out,
            'M': self.M,
            # 'synapse_activation': self.synapse_activation,
            'activation': self.activation,
        })
        return config



class DNMLayer2(Layer):
    def __init__(self, size_out, M, synapse_activation=tf.nn.relu, activation=None, **kwargs):
        super(DNMLayer2, self).__init__(**kwargs)
        self.size_out = size_out
        self.M = M
        self.synapse_activation = synapse_activation
        self.activation = activation

    def build(self, input_shape):
        size_in = input_shape[-1]
        
        # dendritic
        self.W = self.add_weight(shape=(size_in, self.size_out * self.M),
                                 initializer=tf.keras.initializers.RandomUniform(),
                                 trainable=True, name="weights")
        self.b = self.add_weight(shape=(self.size_out * self.M,),
                                 initializer=tf.keras.initializers.Zeros(),
                                 trainable=True, name="biases")
        
        # membrane
        self.DNM_weight = self.add_weight(shape=(self.size_out, self.M),
                                          initializer=tf.keras.initializers.RandomUniform(),
                                          trainable=True, name="dnm_weights")
        
        # Soma
        self.soma_weight = self.add_weight(shape=(self.size_out,),
                                           initializer='random_normal', name="soma_weights",trainable=True)
        self.soma_b = self.add_weight(shape=(self.size_out,),
                                      initializer=tf.keras.initializers.Zeros(),
                                 trainable=True, name="soma_biases")
        self.soma_k = self.add_weight(shape=(), initializer=tf.constant_initializer(0.3),
                                      name="soma_k",trainable=True)
        self.k = self.add_weight(name="k", shape=(), initializer=tf.constant_initializer(0.1), trainable=True)
        
        super(DNMLayer2, self).build(input_shape)

    def call(self, inputs):
        wx_plus_b = tf.multiply(tf.add(tf.matmul(inputs, self.W), self.b), self.k)
        wx_plus_b = self.synapse_activation(wx_plus_b)

        wx_plus_b = tf.reshape(wx_plus_b, [-1, self.size_out, self.M])
        DNM_fc = tf.multiply(wx_plus_b, self.DNM_weight)
        wx_plus_b = tf.reduce_prod(DNM_fc, axis=2)

        wx_plus_b = tf.multiply(tf.nn.relu(tf.multiply(self.soma_weight, wx_plus_b)), self.soma_k)
        if self.activation:
            wx_plus_b = self.activation(wx_plus_b)

        return wx_plus_b
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size_out': self.size_out,
            'M': self.M,
            # 'synapse_activation': self.synapse_activation,
            'activation': self.activation,
        })
        return config


class DNMLayerDense(Layer):
    def __init__(self, size_out, M, synapse_activation=tf.nn.relu, activation=None, **kwargs):
        super(DNMLayerDense, self).__init__(**kwargs)
        self.size_out = size_out
        self.M = M
        self.synapse_activation = synapse_activation
        self.activation = activation
        

    def build(self, input_shape):
        self.dense_layer = tf.keras.layers.Dense(self.size_out * self.M, name='dense_layer') # 'glorot_uniform' #GlorotNormal TruncatedNormal(stddev,GlorotNormal(tf.sqrt(2 / (self.size_out + self.M)))
        self.dnm_weight = self.add_weight(name='dnm_weight', shape=(self.size_out, self.M),
                                          initializer=tf.keras.initializers.GlorotUniform(tf.sqrt(2 / (self.size_out + self.M))), trainable=True)
        self.k = self.add_weight(name="k", shape=(), initializer=tf.constant_initializer(0.1), trainable=True)
    def call(self, inputs):
        fc = self.dense_layer(inputs)
        k_fc = tf.multiply(fc, self.k)
        activation_fc = self.synapse_activation(k_fc)

        reshape_activation_fc = tf.reshape(activation_fc, [-1, self.size_out, self.M])
        dnm_fc = tf.multiply(reshape_activation_fc, self.dnm_weight)
        # dnm_fc = tf.nn.relu(reshape_activation_fc)
        # out = tf.reduce_sum(reshape_activation_fc, axis=2)
        out = tf.reduce_sum(dnm_fc, axis=2)
        if self.activation:
            out = self.activation(out)
        return out
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size_out': self.size_out,
            'M': self.M,
            #'synapse_activation': self.synapse_activation,
            # 'activation': self.activation,
        })
        return config
    

# calculate MAPE
def mean_absolute_percentage_error(real, predict):
    res = 0
    count = 0
    for i in range(len(real)):
        if real[i] != 0:
            res += abs((predict[i]-real[i])/real[i])
            count += 1
    if count == 0:
        return 0  # Avoid division by zero
    return res/count


data = pd.read_csv(r'CLDNM\train_data\IMF2.csv')


power_data = data.iloc[:, -1] 


# 选择特定的列作为特征列，包括 imf1 到 imf11 和 res 列
imf_columns = ['imf1', 'imf2', 'imf3', 'imf4', 'imf5', 'imf6', 'imf7', 'imf8', 'imf9', 'imf10','imf11']
res_column = 'res'  
other_columns = ['Temperature', 'humidity', 'Liquid', 'Wind Speed', 'holiday']
# 创建一个包含选定列的 DataFrame，作为 decompose_data
decompose_data = data[imf_columns + [res_column]+ other_columns]
# decompose_data = pd.concat([data[imf_columns], data[res_column]], axis=1)


# 可以为DataFrame添加列名，如果需要的话
decompose_data.columns = imf_columns + [res_column] + other_columns

# 将 decompose_data 转换为 NumPy 数组
decompose_values = decompose_data.values



print(decompose_data.head())



def TimeSeries(dataset, start_index, history_size, end_index, step,
               target_size, point_time, true):
    data = []  # 保存特征数据
    labels = []  # 保存特征数据对应的标签值

    start_index = start_index + history_size  # 第一次的取值范围[0:start_index]

    # 如果没有指定滑动窗口取到哪个结束，那就取到最后
    if end_index is None:
        # 数据集最后一块是用来作为标签值的，特征不能取到底
        end_index = len(dataset) - target_size

    # 滑动窗口的起始位置到终止位置每次移动一步
    for i in range(start_index, end_index):

        index = range(i - history_size, i, step)  # 第一次相当于range(0, start_index, 6)

        # 根据索引取出所有的特征数据的指定行
        data.append(dataset[index])

        # 用这些特征来预测某一个时间点的值还是未来某一时间段的值
        if point_time is True:  # 预测某一个时间点
            # 预测未来哪个时间点的数据，例如[0:20]的特征数据（20取不到），来预测第20个的标签值
            labels.append(true[i + target_size])

        else:  # 预测未来某一时间区间
            # 例如[0:20]的特征数据（20取不到），来预测[20,20+target_size]数据区间的标签值
            labels.append(true[i:i + target_size])

    labels = np.array(labels)
    # 返回划分好了的时间序列特征及其对应的标签值
    return np.array(data), labels


#划分训练测试集
def get_tain_val_test(serie_data,window_size):
    train_num = int(len(serie_data)*0.5)
    val_num = int(len(serie_data)*0.6)  # 验证集划分
    history_size = window_size  
    target_size =  0 # 预测未来下一个时间点的气温值
    step = 1  # 步长为1取所有的行

    # 求训练集的每个特征列的均值和标准差
    feat_mean = serie_data.mean(axis=0)
    feat_std = serie_data.std(axis=0)

    # 对整个数据集计算标准差
    feat = (serie_data - feat_mean) / feat_std

    # 构造训练集
    x_train, y_train = TimeSeries(dataset=serie_data, start_index=0, history_size=history_size, end_index=train_num,
                                  step=step, target_size=target_size, point_time=True, true=serie_data)
    # 构造验证集
    x_val, y_val = TimeSeries(dataset=serie_data, start_index=train_num, history_size=history_size, end_index=val_num,
                              step=step, target_size=target_size, point_time=True, true=serie_data)
    # 构造测试集
    x_test, y_test =  TimeSeries(dataset=serie_data, start_index=val_num, history_size=history_size, end_index=len(serie_data),
                                  step=step, target_size=target_size, point_time=True, true=serie_data)

    # 查看数据集信息
    print('x_train_shape:', x_train.shape)
    print('y_train_shape:', y_train.shape)
    return x_train,y_train,x_val, y_val,x_test, y_test

import time

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.times.append(epoch_time)
        print(f"Epoch {epoch + 1} time: {epoch_time:.8f} seconds")

def implement_LSTM(X_train, y_train,  X_validate, y_validate, verbose = 1, model_summary = True):
    
    model = Sequential()
    model.add(LSTM(64, input_shape=(window_size, 6)))
    # model.add(Bidirectional(LSTM(64),input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(DNMLayerDense(size_out=1, M=10, synapse_activation=tf.nn.relu, activation=None))
    # model.add(DNMLayer2(size_out=1, M=10, synapse_activation=tf.nn.relu, activation=None))
    #model.add(Dense(1))
    if model_summary:
        model.summary()
    
    # optimizer = adam_v2.Adam(learning_rate=0.01)
    model.compile(loss='mean_squared_error',  metrics=['mae']) #optimizer=optimizer,

    
    time_callback = TimeHistory()


    history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_validate,y_validate),callbacks=[time_callback]) #callbacks=[early_stopping], verbose=1)

    return history,model





window_size = 3
x_train_all,y_train_all,x_val_all, y_val_all,x_test_all, y_test_all = get_tain_val_test(power_data, window_size)
# y_pre_all = model.predict(x_test_all).reshape(-1)



def mean_absolute_error(y_test,y_pre):
    mae = np.sum(np.absolute(y_pre-y_test))/len(y_test)
    return mae
def mean_squared_error(y_test,y_pre):
    mse = np.sum((y_pre-y_test)**2)/len(y_test)
    return mse
def h_mean_absolute_error(y_test,y_pre):
    hmae = mean_absolute_error(y_test,y_pre) / np.mean(y_pre)
    return hmae
def h_mean_squared_error(y_test,y_pre):
    hmse = mean_squared_error(y_test,y_pre) / np.mean(y_pre) ** 2
    return hmse



# 创建一个空的 DataFrame 以存储所有预测值
all_predictions_df = pd.DataFrame()

# 创建空的列表以存储每次循环的指标值
mae_list = []
rmse_list = []
mape_list = []
r_squared_list = []
for i in range(10):
    print('Number of cycles:',i+1)
    
    y_pre_list = []
    y_test_list = []
    for column in decompose_data.columns[:12]:
        serie_data = decompose_data[column]
        # 将IMF列与其他特征列合并
        serie_data = np.column_stack((serie_data, decompose_data[other_columns].values))
        
        print('serie_date.shape', serie_data.shape)
        # 创建 MinMaxScaler
        scaler = MinMaxScaler()

        # 对 serie_data 进行归一化
        serie_data = scaler.fit_transform(serie_data) # .reshape(-1, 1)
        
        print('serie_date.shape', serie_data.shape)
        x_train, y_train, x_val, y_val, x_test, y_test = get_tain_val_test(serie_data,window_size)
        y_train = y_train[:,0].reshape(-1,1)
        print('y_train.shape', y_train.shape)
        y_test = y_test[:,0].reshape(-1,1)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6)) # x_train.shape[1]
        
        x_test = np.reshape(x_test, (x_test.shape[0], x_train.shape[1], 6))
        print('x_train.shape', x_train.shape)

        y_val = y_val[:,0].reshape(-1,1)
        x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 6)) 

       
        # 定义 EarlyStopping 回调，监控验证集上的损失值，当连续 5 个 epoch 损失值不再下降时停止训练
        # early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        history,model = implement_LSTM(x_train, y_train, x_val, y_val)


        # print(early_stopping.stopped_epoch)
        # 保存最佳模型到文件
        # model.save(r'CLDNM\train_data\best_model.h5')

        # 在加载模型之前注册自定义层
        # custom_objects = {'DNMLayer': DNMLayer}

        # 加载最佳模型进行预测
        # best_model = keras.models.load_model(r'CLDNM\train_data\best_model.h5')#, custom_objects = custom_objects)
        y_pre = model.predict(x_test)
        
        y_pre_stacked = concatenate((y_pre, y_pre, y_pre, y_pre, y_pre, y_pre), axis=1)
        y_pre = scaler.inverse_transform(y_pre_stacked)[:,0].reshape(-1,1)
        
        y_test_stacked = concatenate((y_test, y_test, y_test, y_test, y_test, y_test), axis=1)
        y_test = scaler.inverse_transform(y_test_stacked)[:,0].reshape(-1,1)
        y_pre_list.append(y_pre)
        y_test_list.append(y_test)
        

    #此时的预测是对全部分解结果的预测求和
    y_pre_total = np.sum(np.array(y_pre_list),axis = 0).reshape(-1)

    # 计算每次循环的MAE
    mae = mean_absolute_error(y_test_all, y_pre_total)
    mae_list.append(mae)
    
    # 计算每次循环的RMSE
    rmse = np.sqrt(mean_squared_error(y_test_all, y_pre_total))
    rmse_list.append(rmse)
    
    # 计算所有预测结果的MAPE
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # 计算每次循环的MAPE
    mape = mean_absolute_percentage_error(y_test_all, y_pre_total)
    mape_list.append(mape)
    
    # 计算R方
    def r_squared(y_true, y_pred):
        ssr = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ssr / sst)

    # 计算每次循环的R^2
    r2 = r_squared(y_test_all, y_pre_total)
    r_squared_list.append(r2)

    # 创建一个DataFrame来存储反归一化后的预测值
    predictions_df = pd.DataFrame({'Predicted': y_pre_total})

    # 将当前预测值DataFrame追加到all_predictions_df中
    all_predictions_df = pd.concat([all_predictions_df, predictions_df], axis=1)
    



    # 打印结果
    print("MAE{}: {:.4f}".format(i, mae_list[-1]))
    print("RMSE{}: {:.4f}".format(i, rmse_list[-1]))
    print("MAPE{}: {:.4f}%".format(i, mape_list[-1]))
    print("R^2{}: {:.4f}".format(i, r_squared_list[-1]))




# 将DataFrame保存为CSV文件
all_predictions_df.to_csv(r'CLDNM\train_data\predictions.csv', index=False)




# 计算每个指标的平均值
average_mae = np.mean(mae_list)
average_rmse = np.mean(rmse_list)
average_mape = np.mean(mape_list)
average_r_squared = np.mean(r_squared_list)


# 创建一个包含每次循环指标的DataFrame
loop_metrics_df = pd.DataFrame({
    'MAE': mae_list,
    'RMSE': rmse_list,
    'MAPE': mape_list,
    'R^2': r_squared_list
})

# 将每次循环的指标保存到CSV文件的同一个sheet中
loop_metrics_df.to_csv(r'CLDNM\train_data\predictions.csv', mode='a', header=False, index=False)

# 创建一个包含指标平均值的DataFrame
average_metrics_df = pd.DataFrame({
    'Metric': ['Average MAE', 'Average RMSE', 'Average MAPE', 'Average R^2'],
    'Value': [average_mae, average_rmse, average_mape, average_r_squared]
})

# 将指标平均值保存到CSV文件的不同sheet中
average_metrics_df.to_csv(r'CLDNM\train_data\predictions.csv', mode='a', header=False, index=False)

# 打印结果
print("MAE: {:.4f}".format(average_mae))
print("RMSE: {:.4f}".format(average_rmse))
print("MAPE: {:.4f}%".format(average_mape))
print("R^2: {:.4f}".format(average_r_squared))
