# NASA电池数据集加载和预处理模块
# 支持B0005数据集的80%训练20%测试划分

import os
import numpy as np
import scipy.io
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class NASADataLoader:
    """NASA电池数据集加载器"""
    
    def __init__(self, data_path='dataset/'):
        """
        初始化数据加载器
        
        参数:
            data_path: 数据集路径
        """
        self.data_path = data_path
        self.battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
        self.rated_capacity = 2.0  # 额定容量
        
    def convert_to_time(self, hmm):
        """转换时间格式"""
        year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
        return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

    def load_mat(self, matfile):
        """加载mat文件"""
        data = scipy.io.loadmat(matfile)
        filename = matfile.split('/')[-1].split('.')[0]

        col = data[filename]
        col = col[0][0][0][0]
        size = col.shape[0]

        data = []
        for i in range(size):
            k = list(col[i][3][0].dtype.fields.keys())
            d1, d2 = {}, {}
            if str(col[i][0][0]) != 'impedance':
                for j in range(len(k)):
                    t = col[i][3][0][0][j][0]
                    l = [t[m] for m in range(len(t))]
                    d2[k[j]] = l
            d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(self.convert_to_time(col[i][2][0])), d2
            data.append(d1)

        return data

    def get_battery_capacity(self, battery):
        """提取锂电池容量"""
        cycle, capacity = [], []
        i = 1
        for bat in battery:
            if bat['type'] == 'discharge':
                capacity.append(bat['data']['Capacity'][0])
                cycle.append(i)
                i += 1
        return [cycle, capacity]

    def load_battery_data(self, battery_name=None):
        """
        加载电池数据
        
        参数:
            battery_name: 电池名称，如果为None则加载所有电池
            
        返回:
            电池数据字典
        """
        battery_data = {}
        
        # 首先尝试加载预处理的数据
        try:
            battery_data = np.load(os.path.join(self.data_path, 'NASA.npy'), allow_pickle=True).item()
            print("成功加载预处理的数据")
        except:
            # 如果失败，则从mat文件加载
            print("从mat文件加载数据...")
            batteries_to_load = [battery_name] if battery_name else self.battery_list
            
            for name in batteries_to_load:
                print(f'加载数据集 {name}.mat ...')
                try:
                    path = os.path.join(self.data_path, name + '.mat')
                    data = self.load_mat(path)
                    battery_data[name] = self.get_battery_capacity(data)
                except Exception as e:
                    print(f"加载 {name} 失败: {e}")
                    continue
        
        return battery_data

    def prepare_b0005_data(self, train_ratio=0.7):
        """
        准备B0005数据集，按照70%训练30%测试划分
        
        参数:
            train_ratio: 训练集比例
            
        返回:
            train_data, test_data, full_data
        """
        battery_data = self.load_battery_data()
        
        if 'B0005' not in battery_data:
            raise ValueError("B0005数据集未找到")
        
        # 获取B0005的容量数据
        capacity_data = battery_data['B0005'][1]  # [1]是容量数据
        
        # 计算分割点
        split_point = int(len(capacity_data) * train_ratio)
        
        # 划分训练集和测试集
        train_data = capacity_data[:split_point]
        test_data = capacity_data[split_point:]
        
        print(f"B0005数据集总长度: {len(capacity_data)}")
        print(f"训练集长度: {len(train_data)} ({train_ratio*100:.1f}%)")
        print(f"测试集长度: {len(test_data)} ({(1-train_ratio)*100:.1f}%)")
        
        return train_data, test_data, capacity_data


class TimeSeriesPreprocessor:
    """时间序列数据预处理器"""
    
    def __init__(self, window_size=8, normalize=True):
        """
        初始化预处理器
        
        参数:
            window_size: 时间窗口大小
            normalize: 是否归一化
        """
        self.window_size = window_size
        self.normalize = normalize
        self.scaler = MinMaxScaler() if normalize else None
        
    def build_sequences(self, data, window_size=None):
        """
        构建时间序列样本
        
        参数:
            data: 时间序列数据
            window_size: 窗口大小
            
        返回:
            X, y: 特征和标签
        """
        if window_size is None:
            window_size = self.window_size
            
        X, y = [], []
        for i in range(len(data) - window_size):
            features = data[i:i+window_size]
            target = data[i+window_size]
            X.append(features)
            y.append(target)
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def prepare_ml_data(self, train_data, test_data):
        """
        为机器学习模型准备数据
        
        参数:
            train_data: 训练数据
            test_data: 测试数据
            
        返回:
            X_train, y_train, X_test, y_test, scaler
        """
        # 构建训练集序列
        X_train, y_train = self.build_sequences(train_data)
        
        # 为测试集构建序列（使用训练集的最后window_size个点作为初始窗口）
        full_data = list(train_data) + list(test_data)
        X_test, y_test = self.build_sequences(full_data)
        
        # 只保留测试集部分
        X_test = X_test[len(train_data) - self.window_size:]
        y_test = y_test[len(train_data) - self.window_size:]
        
        # 归一化
        if self.normalize and self.scaler is not None:
            # 拟合训练数据
            X_train_reshaped = X_train.reshape(-1, 1)
            y_train_reshaped = y_train.reshape(-1, 1)
            
            # 合并训练数据进行拟合
            train_all = np.vstack([X_train_reshaped, y_train_reshaped])
            self.scaler.fit(train_all)
            
            # 转换数据
            X_train = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
            y_train = self.scaler.transform(y_train_reshaped).flatten()
            
            X_test_reshaped = X_test.reshape(-1, 1)
            y_test_reshaped = y_test.reshape(-1, 1)
            X_test = self.scaler.transform(X_test_reshaped).reshape(X_test.shape)
            y_test = self.scaler.transform(y_test_reshaped).flatten()
        
        return X_train, y_train, X_test, y_test, self.scaler
    
    def prepare_dl_data(self, train_data, test_data):
        """
        为深度学习模型准备数据
        
        参数:
            train_data: 训练数据
            test_data: 测试数据
            
        返回:
            X_train, y_train, X_test, y_test, scaler
        """
        X_train, y_train, X_test, y_test, scaler = self.prepare_ml_data(train_data, test_data)
        
        # 为深度学习模型重塑数据 (samples, timesteps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        return X_train, y_train, X_test, y_test, scaler
    
    def inverse_transform(self, data):
        """反归一化"""
        if self.scaler is not None:
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return self.scaler.inverse_transform(data).flatten()
        return data


def load_and_prepare_data(data_path=None, window_size=8, train_ratio=0.7, normalize=True):
    """
    便捷函数：加载并准备B0005数据
    
    参数:
        data_path: 数据路径（可选）
        window_size: 时间窗口大小
        train_ratio: 训练集比例
        normalize: 是否归一化
        
    返回:
        数据字典包含ML和DL格式的数据
    """
    # 加载数据
    if data_path is not None:
        loader = NASADataLoader(data_path)
    else:
        loader = NASADataLoader()
    train_data, test_data, full_data = loader.prepare_b0005_data(train_ratio)
    
    # 预处理
    preprocessor = TimeSeriesPreprocessor(window_size, normalize)
    
    # 准备ML数据
    X_train_ml, y_train_ml, X_test_ml, y_test_ml, scaler = preprocessor.prepare_ml_data(train_data, test_data)
    
    # 准备DL数据
    X_train_dl, y_train_dl, X_test_dl, y_test_dl, _ = preprocessor.prepare_dl_data(train_data, test_data)
    
    return {
        'ml_data': {
            'X_train': X_train_ml,
            'y_train': y_train_ml,
            'X_test': X_test_ml,
            'y_test': y_test_ml
        },
        'dl_data': {
            'X_train': X_train_dl,
            'y_train': y_train_dl,
            'X_test': X_test_dl,
            'y_test': y_test_dl
        },
        'raw_data': {
            'train_data': train_data,
            'test_data': test_data,
            'full_data': full_data
        },
        'scaler': scaler,
        'preprocessor': preprocessor
    }


if __name__ == "__main__":
    # 测试数据加载
    data_dict = load_and_prepare_data()
    print("数据加载完成！")
    print(f"ML训练集形状: {data_dict['ml_data']['X_train'].shape}")
    print(f"ML测试集形状: {data_dict['ml_data']['X_test'].shape}")
    print(f"DL训练集形状: {data_dict['dl_data']['X_train'].shape}")
    print(f"DL测试集形状: {data_dict['dl_data']['X_test'].shape}")