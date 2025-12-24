import torch
import torch.nn as nn
import torch.optim as optim
from model_MsAutoformer import MsAutoformer_EncoderOnly
from test import test_main
from tool_for_pre import get_parameters, load_data_loaders
from tool_for_train import train_model


def train(args):
    model = MsAutoformer_EncoderOnly(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    train_loader, val_loader = load_data_loaders(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    model_file_path = train_model(model, train_loader, val_loader, criterion, optimizer, 
                                  args.num_epochs, device, args)
    return model_file_path


if __name__ == "__main__":
    args = get_parameters(
        modelname="MsAutoformer", target="LABEL", input_size=9, output_size=4, 
        batch_size=1024, num_epochs=10, learning_rate=5e-4,
        input_directory="data_save/储层流体识别"
    )
    model_file_path = train(args)
    test_main(args, model_file_path=model_file_path)
