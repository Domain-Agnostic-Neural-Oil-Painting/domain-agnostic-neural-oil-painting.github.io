import time
from options.train_options import TrainOptions
from torch.utils.data import DataLoader
from models import create_model
import patches
import torch
import os
import matplotlib.pyplot as plt


def save_networks(model_name, save_dir, gpu_ids, obj):
    """
    Save the network to the disk.

    Parameters:
        model_name (str) -- the name of the model to be saved
        save_dir (str) -- directory to save the model
        gpu_ids (list) -- list of GPU ids
    """
    save_filename = f'{model_name}_updated.pth'
    save_path = os.path.join(save_dir, save_filename)
    
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        torch.save(obj.module.cpu().state_dict(), save_path)  # move to CPU before saving
        obj.cuda(gpu_ids[0])  # move back to GPU after saving
    else:
        torch.save(obj.cpu().state_dict(), save_path)  # save directly if no GPU available


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    isTrain = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印当前使用的设备
    print(f"当前使用的设备: {device}")

    # 输入和输出目录
    input_dir = '../dataset/original/512/test'
    output_dir = '../dataset/new/test'
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        
        # 在每个epoch结束后保存微调后的模型
        checkpoints_dir = os.path.join('checkpoints/painter', input_filename)
        os.makedirs(checkpoints_dir, exist_ok=True)  

        # loss_dir = os.path.join('checkpoints/loss', input_filename)
        # os.makedirs(loss_dir, exist_ok=True)  

        # 从 patches.py 创建patches
        patches_data = patches.create_patches(input_path)
        dataset = patches.PatchDataset(patches_data)
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
        dataset_size = len(dataset)    # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)

        
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        
        total_iters = 0                # the total number of training iterations
        # 在主循环开始之前初始化
        # all_epochs_average_loss = []

        # 进行TTA步骤，使用 n_epochs = tta_steps
        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            
            # 在新epoch开始时加载上一个epoch微调后的模型
            if epoch > opt.epoch_count:  # 从第二个epoch开始
                updated_path = os.path.join(checkpoints_dir, f"model_{input_filename}_epoch_{epoch-1}_updated.pth")
                patches_data = patches.create_patches(input_path, model_path=updated_path)  # 使用更新的模型影响数据生成
                dataset = patches.PatchDataset(patches_data)
                dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
                dataset_size = len(dataset)    # get the number of images in the dataset.
                print('The number of training images = %d' % dataset_size)
            
            model.net_g.train()
            
            # 更新当前epoch
            model.current_epoch = epoch - 1  # 因为epoch从1开始，但我们需要从0开始

            # cumulative_loss = 0
            # batch_count = 0
            # epoch_average_loss = []

            for i, (target_canvas, current_canvas) in enumerate(model.process_batch(patches_data, opt.batch_size)):
                iter_start_time = time.time()  # 每次迭代的计算计时器
                
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                # 更新总迭代次数
                model.current_step = total_iters / opt.batch_size
                
                model.old1 = current_canvas.to(device)
                model.render1 = target_canvas.to(device)

                # print(f'self.render1.shape:{model.render1.shape}')
                # print(f'self.old1.shape:{model.old1.shape}')

                model.old2 = torch.flip(model.old1, [3])
                model.render2 = torch.flip(model.render1, [3])

                model.optimize_parameters()  # 计算损失函数，获取梯度，更新网络权重

                if model.current_epoch <= 1:
                    model.update_learning_rate()

                t_comp = (time.time() - iter_start_time) / opt.batch_size
                t_data = iter_start_time - iter_data_time
                print(f'TTA Step [{epoch}/{opt.n_epochs + opt.n_epochs_decay}], '
                        f'Step [{i+1}/{len(dataloader)}], Loss: {model.loss_all.item():.4f}, '
                        f'Time: {t_comp:.4f}s, Data Loading Time: {t_data:.4f}s')

                iter_data_time = time.time()
            
            print('End of TTA step %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            
            if model.current_epoch > 1:
                model.update_learning_rate()  # update learning rates in the beginning of every epoch.
                
            # 保存当前epoch微调后的模型
            save_networks(f'model_{input_filename}_epoch_{epoch}', checkpoints_dir, [0], model.net_g)
            print(f'Model saved after epoch {epoch} as {os.path.join(checkpoints_dir, f"model_{input_filename}_epoch_{epoch}_updated.pth")}')
        
        # 保存更新后的模型
        save_networks(f'model_{input_filename}', output_dir, [0], model.net_g)
        print(f'Model saved after TTA steps as {os.path.join(output_dir, f"model_{input_filename}_updated.pth")}')
