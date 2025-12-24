import argparse
import os
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

def load_excel_files(directory):
    """
    读取指定目录下的所有xlsx文件并合并成一个DataFrame
    """
    print("开始读取Excel文件...")
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    total_files = len(files)
    data_frames = []
    for i, file in enumerate(files):
        progress = (i + 1) / total_files * 100
        print(f"正在读取文件: {file} (全文件读取进度：{progress:.2f}%)")
        df = pd.read_csv(os.path.join(directory, file))
        data_frames.append(df)
    print("Excel文件读取完成。")
    return pd.concat(data_frames, ignore_index=True)


def create_time_series(data, target_column, sequence_length):
    """
    将DataFrame转换为时序数据
    修复数据泄露问题：使用前N个时间步预测第N+1个时间步
    """
    print("开始转换为时序数据...")
    X, y = [], []
    total_sequences = len(data) - sequence_length  # 减1是为了预测下一个时间步
    for i in range(total_sequences):
        # 使用前N个时间步的特征
        X.append(data.iloc[i:i + sequence_length].drop(target_column, axis=1).values)
        y.append(data.iloc[i + sequence_length-1][target_column])
    print("时序数据转换完成。")
    return np.array(X), np.array(y)


def split_data(X, y, test_size=0.2, val_size=0.2):
    """
    将数据分为训练集、验证集和测试集
    """
    print("开始划分数据集...")
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42)
    # print("数据集划分完成。")
    # return X_train, X_val, X_test, y_train, y_val, y_test

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size + val_size, random_state=42)
    print("数据集划分完成。")
    return X_train, X_val, y_train, y_val


def create_data_loaders(X_train, X_val, y_train, y_val, batch_size=32):
    """
    创建数据加载器
    """
    print("开始创建数据加载器...")
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print("数据加载器创建完成。")

    return train_loader, val_loader


def normalize_and_save(X_train_val, y_train_val, save_dir="data_save/本次数据读取的缓存"):
    """
    对X_train_val和y_train_val进行归一化，并保存归一化器到指定目录。

    :param X_train_val: 输入特征数据，形状为(Batch size, sequence len, features)
    :param y_train_val: 输出标签数据，形状为(Batch size,)
    :param save_dir: 归一化器保存路径
    """
    # 创建保存目录
    print("正在进行归一化...")

    os.makedirs(save_dir, exist_ok=True)

    # Flatten数据以便进行归一化
    X_flattened = X_train_val.reshape(-1, X_train_val.shape[-1])  # 展平到(Batch size * sequence len, features)

    # 初始化MinMaxScaler，进行归一化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # 对特征数据进行归一化
    X_normalized = scaler_X.fit_transform(X_flattened).reshape(X_train_val.shape)

    # 对标签数据进行归一化（如果需要的话）
    y_normalized = scaler_y.fit_transform(y_train_val.reshape(-1, 1))

    # 保存归一化器
    with open(os.path.join(save_dir, "scaler_X.pkl"), 'wb') as f:
        pickle.dump(scaler_X, f)

    with open(os.path.join(save_dir, "scaler_y.pkl"), 'wb') as f:
        pickle.dump(scaler_y, f)

    print("归一化完毕且已保存归一化器...")
    return X_normalized, y_normalized


def normalize_and_load(X_new, y_new, scaler_dir="data_save/本次数据读取的缓存"):
    """
    使用保存的归一化器对新数据进行归一化。

    :param X_new: 新的特征数据，形状为(Batch size, sequence len, features)
    :param y_new: 新的标签数据，形状为(Batch size,)
    :param scaler_dir: 归一化器的保存路径
    :return: 归一化后的X和y
    """
    # 加载已保存的归一化器
    with open(os.path.join(scaler_dir, "scaler_X.pkl"), 'rb') as f:
        scaler_X = pickle.load(f)

    with open(os.path.join(scaler_dir, "scaler_y.pkl"), 'rb') as f:
        scaler_y = pickle.load(f)

    # 对特征数据进行归一化
    X_normalized = scaler_X.transform(X_new.reshape(-1, X_new.shape[-1])).reshape(X_new.shape)

    # 对标签数据进行归一化
    y_normalized = scaler_y.transform(y_new.reshape(-1, 1))

    return X_normalized, y_normalized


def inverse_normalize_and_load(depth, y_normalized, scaler_dir="data_save/本次数据读取的缓存", args=None):
    """
    使用保存的归一化器对预测结果进行反归一化。

    :param X_normalized: 归一化后的特征数据，形状为(Batch size, sequence len, features)
    :param y_normalized: 归一化后的标签数据，形状为(Batch size,)
    :param scaler_dir: 归一化器的保存路径
    :return: 反归一化后的X和y
    """
    # 加载归一化器
    with open(os.path.join(scaler_dir, "scaler_X.pkl"), 'rb') as f:
        scaler_X = pickle.load(f)

    with open(os.path.join(scaler_dir, "scaler_y.pkl"), 'rb') as f:
        scaler_y = pickle.load(f)

    depth_expanded = np.tile(depth[:, np.newaxis], (1, args.input_size))
    # 对特征数据进行反归一化
    X_original = scaler_X.inverse_transform(depth_expanded)
    Depth = X_original[:, 0]
    # 对标签数据进行反归一化
    y_original = scaler_y.inverse_transform(y_normalized.reshape(-1, 1))

    return Depth, y_original


def inverse_normalize_and_load_ture(my_input, y_normalized, scaler_dir="data_save/本次数据读取的缓存", args=None):
    """
    使用保存的归一化器对预测结果进行反归一化。

    :param X_normalized: 归一化后的特征数据，形状为(Batch size, sequence len, features)
    :param y_normalized: 归一化后的标签数据，形状为(Batch size,)
    :param scaler_dir: 归一化器的保存路径
    :return: 反归一化后的X和y
    """
    # 加载归一化器
    with open(os.path.join(scaler_dir, "scaler_X.pkl"), 'rb') as f:
        scaler_X = pickle.load(f)

    with open(os.path.join(scaler_dir, "scaler_y.pkl"), 'rb') as f:
        scaler_y = pickle.load(f)

    # 对特征数据进行反归一化
    X_original = scaler_X.inverse_transform(my_input)
    # 对标签数据进行反归一化
    y_original = scaler_y.inverse_transform(y_normalized.reshape(-1, 1))

    return X_original, y_original




def main(directory, target_column, sequence_length, batch_size=32, normalization_path="data_save/本次数据读取的缓存", args=None):
    print("开始数据处理流程...")
    data_train_val = load_excel_files(os.path.join(directory, "训练集和验证集"))
    X_train_val, y_train_val = create_time_series(data_train_val, target_column, sequence_length)

    # 归一化并保存归一化器
    X_train_val_normalized, y_train_val_normalized = normalize_and_save(X_train_val, y_train_val, save_dir=normalization_path)

    # 划分训练集、验证集和测试集
    X_train, X_val, y_train, y_val = split_data(X_train_val_normalized, y_train_val_normalized)

    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(X_train, X_val, y_train, y_val, batch_size)

    # 打印数据集的形状
    print(f'注意：以下为输入情况：')
    print(f'训练集: X={X_train.shape}, y={y_train.shape}')
    print(f'验证集: X={X_val.shape}, y={y_val.shape}')

    print("数据预处理流程完成")

    return train_loader, val_loader


def save_data_loaders(train_loader, val_loader, save_directory="data_save/本次数据读取的缓存"):
    """
    保存数据加载器到指定目录并存储目录路径
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(os.path.join(save_directory, 'train_loader.pkl'), 'wb') as f:
        pickle.dump(train_loader, f)
    with open(os.path.join(save_directory, 'val_loader.pkl'), 'wb') as f:
        pickle.dump(val_loader, f)


def load_data_loaders(args):
    """
    从存储的路径中加载数据加载器
    """
    # 读取存储的目录路径
    save_directory = os.path.join('data_save', '本次数据读取的缓存', args.input_directory)
    with open(os.path.join(save_directory, 'train_loader.pkl'), 'rb') as f:
        train_loader = pickle.load(f)
    with open(os.path.join(save_directory, 'val_loader.pkl'), 'rb') as f:
        val_loader = pickle.load(f)
    # with open(os.path.join(save_directory, 'test_loader.pkl'), 'rb') as f:
    #     test_loader = pickle.load(f)

    print("数据加载器已从目录加载: ", save_directory)
    return train_loader, val_loader


def parse_int_list(arg):
    return [int(x) for x in arg.split(',')]


def get_parameters(modelname="MsAutoformer", target="LABEL", input_size=9, output_size=4, batch_size=1024, num_epochs=50, learning_rate=5e-4, input_directory="data_save/新数据", hidden_size=8):
    parser = argparse.ArgumentParser(description='训练模型的脚本')
    ## model
    parser.add_argument('--model_name', type=str, default=modelname, help='选择一个模型')
    parser.add_argument('--hidden_size', type=int, default=hidden_size, help='隐藏层的神经元数量')
    parser.add_argument('--num_layers', type=int, default=5, help='层的数量')
    parser.add_argument('--dropout', type=float, default=0.1, help='丢失概率')
    parser.add_argument('--hidden_space', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=2)
    # === Autoformer-specific ===
    parser.add_argument('--e_layers', type=int, default=2, help='Autoformer Encoder 层数')
    parser.add_argument('--d_ff', type=int, default=64, help='前馈层维度')
    parser.add_argument('--moving_avg', type=int, default=24, help='滑动平均窗口大小')
    parser.add_argument('--factor', type=int, default=4, help='AutoCorrelation Top-k采样因子')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数: relu 或 gelu')
    parser.add_argument('--embed', type=str, default='timeF', help='时间特征嵌入类型: timeF 或 fixed')
    parser.add_argument('--freq', type=str, default='h', help='时间频率，例如: h（小时），d（天）')
    parser.add_argument('--seq_len', type=int, default=64, help='时序数据的长度')
    # training
    parser.add_argument('--num_epochs', type=int, default=num_epochs, help='训练的轮数')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='学习率')
    # data
    parser.add_argument('--input_directory', type=str, default=f'{input_directory}', help='输入地址')
    parser.add_argument('--predict_target', type=str, default=f'{target}', help='预测目标')
    parser.add_argument('--input_size', type=int, default=input_size, help='输入特征的维度')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='批次大小')
    parser.add_argument('--output_size', type=int, default=output_size, help='输出特征的维度')
    parser.add_argument('--sequence_length', type=int, default=64, help='时序数据的长度')
    args, unknown = parser.parse_known_args()

    return args
