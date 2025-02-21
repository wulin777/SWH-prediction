import torch
import numpy as np
import random
import tqdm

# 多步的滑动窗口设置
def get_rolling_window(forecasting_length, interval_length, window_length, total_labels):
    # 计算迭代次数
    total_time_steps = total_labels.shape[0]
    required_steps = window_length + interval_length + forecasting_length
    iteration_count = total_time_steps - required_steps + 1

    # 判断是否有足够的数据进行迭代
    if iteration_count <= 0:
        return None  # 如果数据不足，提前退出函数

    # 初始化输出数组
    output_labels = np.zeros((1, forecasting_length))

    # 对每个样本应用滚动窗口
    for index in tqdm.tqdm(range(iteration_count), desc='Signal preparing'):
        new_labels = np.expand_dims(total_labels['signal'].iloc[
                                    index + interval_length + window_length: index + interval_length + window_length + forecasting_length].values,
                                    axis=0)
        output_labels = np.concatenate((output_labels, new_labels), axis=0)

    # 移除初始化时的全零行，并转换为numpy数组
    output_labels = output_labels[1:]

    return output_labels

def get_rolling_window_multistep_3d(forecasting_length, interval_length, window_length, features, labels):

    # 计算迭代次数
    total_time_steps = features.shape[1]
    required_steps = window_length + interval_length + forecasting_length
    iteration_count = total_time_steps - required_steps + 1

    # 判断是否有足够的数据进行迭代
    if iteration_count <= 0:
        return None, None  # 如果数据不足，提前退出函数
    # 确定通道数
    num_channels = features.shape[0]
    # 初始化输出数组
    output_features_list = []
    output_labels_list = []

    # 遍历每个通道
    for channel in range(num_channels):
        channel_features = features[channel].T
        channel_labels = labels[channel].T
        output_features = np.zeros((1, channel_features.shape[0], window_length))
        output_labels = np.zeros((1, 1,forecasting_length))

        range_end = channel_features.shape[1] - interval_length - window_length - forecasting_length + 1
        # 对每个通道应用滚动窗口
        for index in tqdm.tqdm(range(0, channel_features.shape[1] - interval_length - window_length - forecasting_length + 1), desc=f'Channel {channel+1} data preparing'):
            new_features = np.expand_dims(channel_features[:,index:index + window_length], axis=0)
            new_labels = np.expand_dims(channel_labels[:,index + interval_length + window_length: index + interval_length + window_length + forecasting_length], axis=0)
            output_features = np.concatenate((output_features, new_features), axis=0)
            output_labels = np.concatenate((output_labels, new_labels), axis=0)

        # 移除初始化时的全零行，并添加到列表
        output_features_list.append(output_features[1:])
        output_labels_list.append(output_labels[1:])

    # 将列表转换为numpy数组
    output_features = np.array(output_features_list)
    output_labels = np.array(output_labels_list)

    return torch.from_numpy(output_features), torch.from_numpy(output_labels)

def tgt_creator(features, input_len, output_len):
    if input_len > output_len:
        tgt = features[:,:, :, -1-output_len:-1]
    else:
        tgt = torch.cat((torch.zeros((features.shape[0], features.shape[1], output_len-features.shape[2])), features[:,:, :, :]), dim=2)
    return tgt
'''
# 创建一个随机特征数组 (9, 100, 6)
features = np.random.rand(9, 100, 6)

# 创建一个随机标签数组，为简单起见，这里只取特征的最后一列
labels = np.random.rand(9, 100,1)

# 定义参数
forecasting_length = 12
interval_length = 0
window_length = 48

# 调用函数
features_tensor, labels_tensor = get_rolling_window_multistep_3d(forecasting_length, interval_length, window_length, features, labels)

print(features_tensor.shape)
print(labels_tensor.shape)
'''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False