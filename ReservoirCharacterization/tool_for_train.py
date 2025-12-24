import os
from datetime import datetime
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
from torch import optim
import shutil
from tool_for_test import print_log


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, args):
    train_losses = []
    val_losses = []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    
    base_dir = os.path.join("models_save", args.model_name + datetime.now().strftime("--%d--%H--%M--%S"))
    os.makedirs(base_dir, exist_ok=True)
    save_path = base_dir
    final_model_file_path = os.path.join(save_path, 'epoch_last.pth')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            targets = (targets * 3).long()
            loss = criterion(outputs, targets.squeeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                targets = (targets * 3).long()
                loss = criterion(outputs, targets.squeeze(1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        if (epoch + 1) % 25 == 0:
            model_file_path = os.path.join(save_path, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_file_path)

    print("训练完成")
    torch.save(model.state_dict(), final_model_file_path)
    print(f"最终模型已保存至 {final_model_file_path}")

    # 绘制损失图并保存
    # 配置字体
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['axes.titlesize'] = 20  # 图标题字体大小
    mpl.rcParams['axes.labelsize'] = 16  # x轴、y轴标签字体大小
    mpl.rcParams['xtick.labelsize'] = 14  # x轴刻度字体大小
    mpl.rcParams['ytick.labelsize'] = 14  # y轴刻度字体大小
    mpl.rcParams['legend.fontsize'] = 14  # 图例字体大小
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='#0077a3', linestyle='-', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='#ff4c4c', linestyle='-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_file_path = os.path.join(save_path, 'loss_plot.png')
    plt.savefig(loss_plot_file_path)  # 保存图像到指定目录
    print(f"损失图已保存至 {loss_plot_file_path}")
    return final_model_file_path
