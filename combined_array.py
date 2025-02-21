import pandas as pd
import numpy as np
from vmdpy import u

# 加载数据
file_path_data = './data.xlsx'
df_data = pd.read_excel(file_path_data, header=0)  # 确保正确处理标题行
print("Length of original data:", len(df_data))
# 加载VMD处理后的数据
u_shape = u.shape  # u是从VMD函数返回的分解模态信号数组，形状应为 [9, N]

# 创建一个空的三维数组
combined_array = np.empty((4106, 6, u_shape[0]))

# 遍历每一个模态
for i in range(u_shape[0]):
    # 复制原始数据的前五列
    temp_data = df_data.iloc[:, :5].copy()
    # 将u中的模态信号添加为新的第六列
    temp_data['new_col'] = u[i, :]
    # 插入到三维数组中
    combined_array[:, :, i] = temp_data.to_numpy()

    transposed_array = combined_array.transpose(2, 0, 1)
np.save('data_array.npy', transposed_array)
# 现在 combined_array 就是你需要的 16760x6x9 的三维数组
print("Array shape:", transposed_array.shape)
print("Array dtype:", transposed_array.dtype)
# 显示第一个模态信号的前5行数据



