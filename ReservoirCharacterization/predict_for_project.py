import os
from datetime import datetime
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch

from model_MsAutoformer import MsAutoformer_EncoderOnly
from tool_for_pre import get_parameters, create_time_series, normalize_and_load


def get_latest_model(model_dir):
    """获取指定目录下最新的模型文件"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError(f"在 {model_dir} 中未找到 .pth 文件")
    
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_model = os.path.join(model_dir, model_files[0])
    print(f"使用最新模型: {latest_model}")
    return latest_model


def predict_main(args, model_file_path=None):
    if model_file_path is None:
        parts = os.path.normpath(args.input_directory).split(os.sep)
        data_name = parts[-1]
        model_base_dir = os.path.join("models_save", data_name)
        
        if not os.path.exists(model_base_dir):
            raise FileNotFoundError(f"模型基础目录不存在: {model_base_dir}")
        
        model_dirs = [d for d in os.listdir(model_base_dir) 
                     if os.path.isdir(os.path.join(model_base_dir, d)) and args.model_name in d]
        
        if not model_dirs:
            raise FileNotFoundError(f"未找到 {args.model_name} 模型目录")
        
        model_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(model_base_dir, x)), reverse=True)
        latest_model_dir = os.path.join(model_base_dir, model_dirs[0])
        model_file_path = get_latest_model(latest_model_dir)
    
    input_directory = os.path.join(args.input_directory, "训练集和验证集")
    excel_files = read_excel_files(input_directory)
    current_time = datetime.now()
    formatted_time = current_time.strftime("--%H--%M--")

    for file_path in excel_files:
        print(f"处理文件: {file_path}")
        last_directory = os.path.basename(file_path)
        data = pd.read_csv(file_path)
        
        X_test, y_test = create_time_series(data, args.predict_target, args.sequence_length)
        X_new_normalized, y_new_normalized = normalize_and_load(
            X_test, y_test, 
            scaler_dir=os.path.join('data_save', '本次数据读取的缓存', args.input_directory)
        )

        test_dataset = TensorDataset(
            torch.tensor(X_new_normalized, dtype=torch.float32),
            torch.tensor(y_new_normalized, dtype=torch.float32)
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        test(args, test_loader, formatted_time, last_directory, model_file_path)


def test(args, test_loader, formatted_time, well_name, model_file_path):
    model = MsAutoformer_EncoderOnly(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(model_file_path))
    model.eval()
    print("模型已加载")

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    model_dir = os.path.dirname(model_file_path)
    model_dir = os.path.join(model_dir, f"预测结果-{formatted_time}")
    os.makedirs(model_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'True': all_targets.flatten(),
        'Predicted': all_predictions.flatten()
    })
    results_df.to_csv(os.path.join(model_dir, f"{well_name}_predictions.csv"), index=False)
    print(f"预测结果已保存")


def read_excel_files(directory):
    excel_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            excel_files.append(os.path.join(directory, filename))
    return excel_files


if __name__ == "__main__":
    args = get_parameters(
        modelname="MsAutoformer", target="LABEL", input_size=9, output_size=4, 
        batch_size=1024, num_epochs=500, learning_rate=5e-4, 
        input_directory="data_save/储层流体识别"
    )
    predict_main(args)
