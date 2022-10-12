from __future__ import print_function, division
import os
import shutil

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision

from transfer_learning_segmentation.models.core_human_seg import CoreHumanSegModel, create_seg_model
from transfer_learning_segmentation.models.segmentation_model_1 import CoreSegModel


def save_checkpoint(model, optimizer, lr_scheduler, epoch, path, iou_res, bce_res):
    snapshot = dict(
        optimizer=optimizer.state_dict(),
        lr_scheduler=lr_scheduler.state_dict(),
    )
    snapshot["epoch"] = epoch
    snapshot["iou_res"] = iou_res
    snapshot["bce_res"] = bce_res
    torch.save(snapshot, path)
    torch.save(model.state_dict(), path[:-4]+'_model.pth')


def load_model_state(configs, log_dir, device):
    path = os.path.join(log_dir, 'checkpoint_best.pth')
    model_class = configs['model_class']

    if os.path.exists(os.path.join(log_dir, 'checkpoint_best.pth')):
        innit = False
        model_path_load = path[:-4] + '_model.pth'

    else:
        innit = True
        iou_res = 0.0
        start_epoch = 0
        model_path_load = configs['backbone_model_path']
        print()
        print('Innit the model from ', model_path_load)
        print()

    human_seg_model = create_seg_model(model_class, model_path_load, innit, device)
    model = CoreSegModel(human_seg_model)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=configs["learning_rate"], momentum=0.9)
    lr_schedule = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    if not innit:
        loaded = torch.load(path)

        optimizer.load_state_dict(loaded['optimizer'])
        lr_schedule.load_state_dict(loaded['lr_scheduler'])
        start_epoch = loaded["epoch"]
        if "iou_res" in loaded.keys():
            iou_res = loaded["iou_res"]
            bce_res = loaded["bce_res"]

        print()
        print('Loaded model from checkpoint: ', path)
        print('Last learning rate ', lr_schedule.get_last_lr()[0])
        if "iou_res" in loaded.keys():
            print("IOU (val) result %.4f" % iou_res.numpy())
            print("BCE (val) result %.4f" % bce_res)

    return model, optimizer, lr_schedule, start_epoch, iou_res


def summary_images(x, y, y_, step, writer, backbone_out=None):
    x_norm = x + abs(x.min())
    x = x_norm / x_norm.max()
    y = torch.concat([y, y, y], 1)
    y_ = torch.unsqueeze(y_, 1)
    y_ = torch.concat([y_, y_, y_], 1)

    print('Summary images ')
    print('y ', y.max(), y.min())
    print('y_ ', y_.max(), y_.min())

    x_im = torchvision.utils.make_grid(x)
    y_im = torchvision.utils.make_grid(y)
    y__im = torchvision.utils.make_grid(y_)

    writer.add_image(f'images', x_im, global_step=step)
    writer.add_image(f'pred', y__im, global_step=step)
    writer.add_image(f'truth', y_im, global_step=step)
    if backbone_out is not None:
        backbone_out_im = torch.concat([backbone_out, backbone_out, backbone_out], 1)
        backbone_out_im = torchvision.utils.make_grid(backbone_out_im)
        writer.add_image(f'backbone_out', backbone_out_im, global_step=step)
