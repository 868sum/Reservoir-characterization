import os
from tool_for_pre import save_data_loaders, main, get_parameters


def data_pre_process(args):
    directory = args.input_directory
    target_column = args.predict_target
    sequence_length = args.sequence_length
    batch_size = args.batch_size

    save_directory = os.path.join('data_save', '本次数据读取的缓存', args.input_directory)
    train_loader, val_loader = main(directory, target_column, sequence_length, batch_size, save_directory, args=args)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_data_loaders(train_loader, val_loader, save_directory=save_directory)


if __name__ == "__main__":
    args = get_parameters(
        modelname="MsAutoformer", target="LABEL", input_size=9, output_size=4, 
        batch_size=1024, num_epochs=500, learning_rate=5e-4, 
        input_directory="data_save/储层流体识别"
    )
    data_pre_process(args)
