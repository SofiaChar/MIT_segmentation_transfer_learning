from __future__ import print_function, division
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import os
from torchinfo import summary
from transfer_learning_segmentation.builds.build_4 import build_4_config
from transfer_learning_segmentation.builds.build_5 import build_5_config
from transfer_learning_segmentation.logger import load_model_state
from transfer_learning_segmentation.segmentation_dataset import construct_dataset, get_dataloaders
from transfer_learning_segmentation.trainer import train_model

cudnn.benchmark = True
plt.ion()


def main(configs):
    DEVICE = configs['device']
    torch.cuda.set_device(DEVICE)

    log_dir = os.path.join(configs['log_dir'], configs['build_name'])
    training_dataset = construct_dataset(configs['train_paths'],  DEVICE)
    val_dataset = construct_dataset(configs['val_paths'], DEVICE)

    dataloaders, dataset_sizes = get_dataloaders(training_dataset, val_dataset, configs)
    print('dataset_sizes ', dataset_sizes)

    # model_conv = model(configs['backbone_model_path'], device=DEVICE)
    model_conv, optimizer, lr_schedule, start_epoch, best_acc = load_model_state(configs, log_dir, DEVICE, True)

    print('torch.cuda.current_device()', torch.cuda.current_device())

    summary(model_conv, (configs["batch_size"], 3, 512, 512))

    writer = SummaryWriter(log_dir=log_dir)

    model_conv = train_model(model_conv, configs, optimizer,
                             lr_schedule, dataloaders, dataset_sizes, writer=writer, start_epoch=start_epoch, best_acc=best_acc, device=DEVICE)


if __name__ == "__main__":

    main(configs=build_5_config)
