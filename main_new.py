from math import sqrt
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from src.convmodels import *
from src.rolling import *
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import random
from src.early_stopping import *  # 导入早停类

# 非模型相关参数定义
seed = random.randint(1, 1000)
setup_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 训练,如果NVIDIA GPU已配置，会自动使用GPU训练否则用cpu
train_ratio = 0.7  # 训练集比例
val_ratio = 0.15  # 验证集比例
test_ratio = 0.15  # 测试集比例
loss_function = 'MSE'  # 损失函数MSE
learning_rate = 0.01  # 基础学习率
weight_decay = 0.001  # 权重衰减系数
momentum = 0.7  # SGD动量

# 模型相关参数定义
batch_size = 64  # 批大小，若用CPU，建议为1
in_channels = 9
input_length = 48
output_length = 48  # 预测数据长度
input_features_enc = 6  # transformer编码器输入数据特征数  海浪参数
input_features_dec = 6  # transformer解码器输入数据特征数
nheads_enc = 4  # 编码器多头注意力机制head数量
nheads_dec = 4  # 解码器多头注意力机制head数量
embedding_features = 64  # 编码器、解码器输入数据编码维度
output_features = 1  # 预测的输出维度，即SWH
dim_feedforward_enc = 1024  # 编码器前向传播维度
dim_feedforward_dec = 1024  # 解码器前向传播维度
n_enc_layer = 4  # 编码器层数
n_dec_layer = 4  # 解码器层数
features_num = 6

# 数据读取
scalar = True  # 归一化
scalar_contain_labels = True  # 归一化过程是否包含目标值的历史数据
# 多步，单步标签
if output_length > 1:
    forecasting_model = 'multi_steps'
else:
    forecasting_model = 'one_steps'

# 假设文件位于子文件夹"data"中
file_path = './combined_array.npy'
features_ = np.load(file_path)  # 为实际读取到的所有数据
print("features_: ", features_.shape)  # 打印形状

# labels_ = df[target_value].values
labels_ = features_[:, :, -1]  # labels 的形状将为 (9, 100)
data_path = r'.\数据集\VMD\ASWH.xlsx'
total_labels = pd.read_excel(data_path)

original_channels = features_.shape[0]
original_sequence = features_.shape[1]
original_features = features_.shape[2]

features_reshaped = features_.reshape(-1, features_.shape[-1])  # 展平前两个维度
# 初步划分训练集、验证集、测试集
split_train_val, split_val_test = int(len(features_reshaped) * train_ratio), \
                                  int(len(features_reshaped) * train_ratio) + int(len(features_reshaped) * val_ratio)
split_train_val1, split_val_test1 = int(len(total_labels) * train_ratio), \
                                    int(len(total_labels) * train_ratio) + int(len(total_labels) * val_ratio)

train_features_ = features_reshaped[:split_train_val]
val_test_features_ = features_reshaped[split_train_val:]

features_ = np.vstack([train_features_, val_test_features_])
features_ = features_.reshape(original_channels, original_sequence, original_features)  # 恢复原始维度

# 应用相同的滑动窗口逻辑到 total_label  #+1
total_labels = get_rolling_window(output_length, 0, input_length, total_labels)
total_labels = torch.from_numpy(total_labels)
features, labels = get_rolling_window_multistep_3d(output_length, 0, input_length,
                                                   features_, np.expand_dims(labels_, -1))

# 构建数据集
labels = torch.squeeze(labels, dim=1)
features = features.to(torch.float32)
labels = labels.to(torch.float32)
total_labels = total_labels.unsqueeze(-1)
total_labels = total_labels.to(torch.float32)  # +1
split_train_val, split_val_test = int(features.shape[1] * train_ratio), int(features.shape[1] * train_ratio) + int(
    features.shape[1] * val_ratio)

train_features, train_labels = features[:, :split_train_val], labels[:, :split_train_val]
train_features = train_features.permute(1, 0, 3, 2)
train_labels = train_labels.permute(1, 0, 3, 2)

val_features, val_labels = features[:, split_train_val:split_val_test], labels[:, split_train_val:split_val_test]
val_features = val_features.permute(1, 0, 3, 2)
val_labels = val_labels.permute(1, 0, 3, 2)

test_features, test_labels = features[:, split_val_test:], labels[:, split_val_test:]
test_features = test_features.permute(1, 0, 3, 2)
test_labels = test_labels.permute(1, 0, 3, 2)

# 划分数据集  +1
split_train_val1, split_val_test1 = int(len(total_labels) * train_ratio), int(len(total_labels) * train_ratio) + int(
    len(total_labels) * val_ratio)
train_total_labels = total_labels[:split_train_val1]
val_total_labels = total_labels[split_train_val1:split_val_test1]
test_total_labels = total_labels[split_val_test1:]

# 数据管道构建
train_Datasets = TensorDataset(train_features.to(device), train_labels.to(device), train_total_labels.to(device))
train_Loader = DataLoader(batch_size=batch_size, dataset=train_Datasets)

val_Datasets = TensorDataset(val_features.to(device), val_labels.to(device), val_total_labels.to(device))
val_Loader = DataLoader(batch_size=batch_size, dataset=val_Datasets)

test_Datasets = TensorDataset(test_features.to(device), test_labels.to(device), test_total_labels.to(device))
test_Loader = DataLoader(batch_size=batch_size, dataset=test_Datasets)

# 模型定义
model = TransformerDMS(in_channels, input_features_enc, input_features_dec, nheads_enc, nheads_dec,
                       embedding_features, output_features, dim_feedforward_enc, dim_feedforward_dec, n_enc_layer,
                       n_dec_layer, input_length, output_length)
# 模型调用
model.to(device)
if loss_function == 'MSE':
    loss_func = nn.MSELoss(reduction='mean')

# 训练代数定义
epochs = 8

# 优化器定义，学习率衰减定义
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                            momentum=momentum)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs // 3, eta_min=0.00001)

# 初始化早停
early_stopping = EarlyStopping(
    patience=5,  # 调整耐心值
    verbose=True,
    path=f'./weighting/model_Parformer_weights_As_{output_length}'
)

# 训练及验证循环
print("——————————————————————Training Starts——————————————————————")
for epoch in range(epochs):
    # 训练
    model.train()
    train_loss_sum = 0
    step = 1
    for step, (feature_, label_, total_labels) in enumerate(train_Loader):
        optimizer.zero_grad()
        prediction, combined_output = model(feature_)
        loss_c = loss_func(prediction, label_)
        loss_a = loss_func(combined_output, total_labels)  # 新增加的整体预测损失
        loss = loss_c + loss_a  # 总损失
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
    print("nepochs = " + str(epoch))
    print('train_loss = ' + str(train_loss_sum))

    # 验证
    model.eval()
    val_loss_sum = 0
    combined_val_loss_sum = 0

    for val_step, (feature_, label_, total_labels) in enumerate(val_Loader):
        with torch.no_grad():
            prediction, combined_output = model(feature_)
            val_loss_c = loss_func(prediction, label_)
            val_loss_a = loss_func(combined_output, total_labels)
            val_loss_sum += val_loss_c.item()
            combined_val_loss_sum += val_loss_a.item()

    # 计算平均组合验证损失
    combined_val_loss_avg = combined_val_loss_sum / len(val_Loader)

    # 应用早停策略
    early_stopping(combined_val_loss_avg, model)
    if early_stopping.early_stop:
        print("== 触发早停，终止训练 ==")
        break

    print(f'Epoch {epoch}:')
    print(f'训练损失: {train_loss_sum:.4f}')
    print(f'验证损失: {val_loss_sum / len(val_Loader):.4f}')
    print(f'组合验证损失: {combined_val_loss_avg:.4f}\n')

print("——————————————————————Training Ends——————————————————————")

# 测试集预测
model.load_state_dict(torch.load(f'./weighting/model_Parformer_weights_A_{output_length}'))  # 调用权重
test_loss_sum = 0

# 测试集inference
print("——————————————————————Testing Starts——————————————————————")
test_loss_sum = 0
combined_test_loss_sum = 0  # 用于跟踪 combined_output 的测试损失

for step, (feature_, label_, total_labels) in enumerate(test_Loader):
    with torch.no_grad():
        prediction, combined_output = model(feature_)  
        if step == 0:
            pre_array = prediction.cpu()
            combined_array = combined_output.cpu() 
            label_array = label_.cpu()
            total_labels_array = total_labels.cpu()  

            loss_c = loss_func(prediction, label_)
            loss_a = loss_func(combined_output, total_labels) 
            test_loss_sum += loss_c.item()
            combined_test_loss_sum += loss_a.item()  
        else:
            pre_array = np.vstack((pre_array, prediction.cpu()))
            combined_array = np.vstack((combined_array, combined_output.cpu()))  
            label_array = np.vstack((label_array, label_.cpu()))
            total_labels_array = np.vstack((total_labels_array, total_labels.cpu()))  

            loss_c = loss_func(prediction, label_)
            loss_a = loss_func(combined_output, total_labels)
            test_loss_sum += loss_c.item()
            combined_test_loss_sum += loss_a.item()

print("Test loss = " + str(test_loss_sum))
print("Combined test loss = " + str(combined_test_loss_sum))  # 打印 combined_output 的总损失
print("——————————————————————Testing Ends——————————————————————")

pre_array = pre_array.squeeze()
test_labels = label_array.squeeze()
combined_array = combined_array.squeeze()
total_labels_array = total_labels_array.squeeze()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 计算 MSE
mse = mean_squared_error(total_labels_array, combined_array)
print("均方误差 (MSE):", mse)

# 计算 MAE
mae = mean_absolute_error(total_labels_array, combined_array)
print("平均绝对误差 (MAE):", mae)

# 计算 R2
r2 = r2_score(total_labels_array, combined_array)
print("决定系数 (R2):", r2)

epsilon = 1e-8
MARE = np.mean(np.abs((total_labels_array - combined_array) / (total_labels_array + epsilon))) * 100
print(f"Relative Error (RE): {MARE:.2f}%")















