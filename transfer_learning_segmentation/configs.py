import torch


configs = {
    "train_paths": {
        # 'supervisely': '/mnt_sda/ML/datasets/semantic_segmentation/human_segmetation/supervisely_person_dataset',
        # 'crowd_instance': '/mnt_sda/ML/datasets/semantic_segmentation/human_segmetation/crowd_instance_human/instance-level_human_parsing/Validation',
        # 'portrait_matting': '/mnt_sda/ML/datasets/semantic_segmentation/human_segmetation/Human_portrait_matting/hk/training',
        # # 'fashion':'/mnt_sda/ML/datasets/semantic_segmentation/human_segmetation/human_fashion_dataset/humanparsing', DISLIKE
        # 'similars':'/mnt_sda/ML/datasets/semantic_segmentation/human_segmetation/human_dataset_similars',
        'ade20k_images': '/mnt_sda/ML/datasets/ADE20K_2021_17_01/images_detectron2/training',

        'ade20k_masks':'/mnt_sda/ML/datasets/ADE20K_2021_17_01/annotations_detectron2/training',

    },
    "val_paths": {
        # 'human_290': '/mnt_sda/ML/datasets/semantic_segmentation/human_segmetation/human_290',
        'images': '/mnt_sda/ML/datasets/semantic_segmentation/sky_segmentation/sky/data',

        'masks':'/mnt_sda/ML/datasets/semantic_segmentation/sky_segmentation/sky/groundtruth',

    },

    "backbone_model_path": '/mnt_sda/ML/Sofia/mit_segmentation/ckpt/ade20k-resnet18dilated-c1_deepsup.pth',
    "log_dir": './runs',
    "batch_size": 8,
}

ofi_trainer_config = {
    "learning_rate": 0.0001,
    "epochs": 20000,
    "steps_per_epoch": 12,
    "criterion": torch.nn.BCELoss()
}

# data_transforms = {'train_transforms': ComposeDouble([
#     # FunctionWrapperDouble(random_flip, input=True, target=True, ndim_spatial=3),
#     # FunctionWrapperDouble(center_crop_to_size, input=True, target=True, size=(3, 512, 512)),
#     FunctionWrapperDouble(normalize, input=True, target=True, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ]),
#     'val_transforms': ComposeDouble([
#         FunctionWrapperDouble(normalize, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])
# }

configs.update(ofi_trainer_config)
# configs.update(data_transforms)
