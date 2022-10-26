from __future__ import print_function, division
import matplotlib.pyplot as plt
import logging
import os
import shutil

from sklearn.metrics import jaccard_score as jsc

import torch
import copy
from tqdm import tqdm
import numpy as np
from torchmetrics import JaccardIndex
from torchmetrics.functional import jaccard_index

from transfer_learning_segmentation.logger import summary_images, save_checkpoint


def train_model(model, configs, optimizer, scheduler, dataloaders, dataset_sizes, writer, start_epoch, best_acc=0.0, device = "cuda:0"):
    num_epochs = configs['epochs']
    criterion = configs['criterion']
    jaccard = JaccardIndex(num_classes=2)

    generators = {'train': iter(dataloaders['train']), 'val': iter(dataloaders['val'])}

    best_acc = best_acc
    # best_model = copy.deepcopy(model)

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                num_steps = configs['steps_per_epoch']
                # num_steps = 10

            else:
                num_steps = dataset_sizes['val']

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            iou_res = 0.0

            for step in tqdm(range(num_steps)):

                try:
                    inputs, labels = next(generators[phase])

                    # if phase == 'train':
                    #     while True:
                    #         inputs, labels = next(generators[phase])
                    #         print('labels.mean()', labels.mean())
                    #         if labels.mean() != 1.:
                    #             break
                    # if phase == 'val':
                    #     inputs, labels = next(generators[phase])
                except StopIteration:
                    generators[phase] = iter(dataloaders[phase])
                    inputs, labels = next(generators[phase])

                # print(inputs.shape)
                # print(labels.shape)
                # for i in range(labels.shape[0]):
                #     plt.imshow(labels[i,0,:,:].cpu().numpy())
                #     plt.show()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, backbone_outputs = model(inputs)

                    backbone_preds = torch.squeeze(backbone_outputs, 1)
                    backbone_preds = torch.clip(backbone_outputs / backbone_preds.max(), 0, 1)

                    preds = torch.squeeze(outputs, 1).detach()

                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                iou = jaccard_index(torch.tensor(np.expand_dims(preds.cpu().numpy(), axis=0)),
                                    torch.tensor(np.uint8(np.expand_dims(labels.cpu().numpy(), axis=0))), num_classes=2, threshold=0.4)

                iou_res += iou
                print('Curr iou ', iou, iou_res)

                if phase == 'val' and (step % 15 == 0):

                    if step == 0:
                        inputs_to_show = inputs
                        labels_to_show = labels
                        preds_to_show = preds
                        backbone_to_show = backbone_preds
                    else:
                        inputs_to_show = torch.concat([inputs_to_show, inputs], 0)
                        labels_to_show = torch.concat([labels_to_show, labels], 0)
                        preds_to_show = torch.concat([preds_to_show, preds], 0)
                        backbone_to_show = torch.concat([backbone_to_show, backbone_preds], 0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / num_steps
            epoch_acc = iou_res / num_steps

            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"IOU/{phase}", epoch_acc, epoch)

            print(f'{phase} Loss: {epoch_loss:.4f}')
            print(f'{phase} IOU: {epoch_acc:.4f}')
            print(f'{phase} BEST IOU: {best_acc:.4f}')


            # save best model
            if phase == 'val':
                summary_images(inputs_to_show, labels_to_show, preds_to_show, epoch, writer, backbone_to_show)

                if epoch_acc > best_acc:
                    print('SAVE CHECKPOINT BEST, curr_best_acc, new_best_acc', best_acc, epoch_acc)
                    best_acc = epoch_acc
                    # best_model = copy.deepcopy(model)
                save_checkpoint(model, optimizer, scheduler, epoch,
                                str(writer.get_logdir()) + f'/checkpoint_best.pth', best_acc, epoch_loss)


        print()
        if epoch % 2 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, str(writer.get_logdir()) + f'/checkpoint_{epoch}.pth',
                            epoch_acc, epoch_loss)
            print('Saved model checkpoint at ', str(writer.get_logdir()) + f'/checkpoint_{epoch}.pth')

    return model

