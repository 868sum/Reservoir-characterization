import os
from datetime import datetime
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from model_MsAutoformer import MsAutoformer_EncoderOnly
from tool_for_pre import create_time_series, normalize_and_load


def test_main(args, model_file_path):
    input_directory = os.path.join(args.input_directory, "测试集")
    if not os.path.exists(input_directory):
        print(f"测试集目录不存在: {input_directory}，跳过测试")
        return
    
    excel_files = read_excel_files(input_directory)
    if not excel_files:
        print(f"未找到测试文件，跳过测试")
        return
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("--%H--%M--")
    
    for file_path in excel_files:
        print(f"处理文件: {file_path}")
        last_directory = os.path.basename(file_path)
        data = pd.read_csv(file_path)
        
        X_test, y_test = create_time_series(data, args.predict_target, args.sequence_length)
        if len(X_test) == 0:
            print(f"警告: {file_path} 生成的时序数据为空，跳过")
            continue
        
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



def evaluate_classification_from_onehot(all_targets, all_predictions, class_names, save_path, prefix="", model_name="Model"):
    """分类评估：支持 one-hot 编码的标签和预测结果"""
    os.makedirs(save_path, exist_ok=True)

    true_labels = np.argmax(all_targets, axis=1)
    pred_labels = np.argmax(all_predictions, axis=1)

    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [acc, precision, recall, f1]
    })
    metrics_df.to_excel(os.path.join(save_path, f"{prefix}classification_metrics.xlsx"), index=False)

    report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_excel(os.path.join(save_path, f"{prefix}classification_report.xlsx"))

    cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm * 100, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Percentage (%)'},
                annot_kws={'size': 20})
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.title("Normalized Confusion Matrix (%)", fontsize=18)
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(rotation=90, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{prefix}confusion_matrix.png"))
    plt.close()

    class_precisions = report_df.loc[class_names, 'precision'].values
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_names, y=class_precisions, hue=class_names, palette="viridis", legend=False)
    plt.ylim(0, 1)
    plt.ylabel("Precision", fontsize=12)
    plt.xlabel("Class", fontsize=12)
    plt.title("Per-Class Precision", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{prefix}class_precision_bar.png"))
    plt.close()

    class_support = [report_df.loc[name, 'support'] for name in class_names]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_names, y=class_support, hue=class_names, palette="Set2", legend=False)
    plt.ylabel("Sample Count", fontsize=12)
    plt.xlabel("Class", fontsize=12)
    plt.title("Support per Class", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{prefix}class_support_bar.png"))
    plt.close()


def test(args, test_loader, formatted_time, well_name, model_file_path):
    model = MsAutoformer_EncoderOnly(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(model_file_path))
    model.eval()
    print("模型已加载")

    all_predictions = []
    all_targets = []
    print("开始测试模型")
    with torch.no_grad():
        for inputs, targets in test_loader:
            num_classes = 4
            targets = (targets * 3).long()
            targets = F.one_hot(targets, num_classes=num_classes).float()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.squeeze(1).cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    model_dir = os.path.dirname(model_file_path)
    model_dir = os.path.join(model_dir, f"测试结果-{formatted_time}")

    evaluate_classification_from_onehot(
        all_targets=all_targets,
        all_predictions=all_predictions,
        class_names=["Non-Reservoir", "Dry Layer", "Water Layer", "Oil Layer"],
        save_path=model_dir,
        prefix=f"{well_name}__",
        model_name=args.model_name
    )


def read_excel_files(directory):
    excel_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            excel_files.append(os.path.join(directory, filename))
    return excel_files
