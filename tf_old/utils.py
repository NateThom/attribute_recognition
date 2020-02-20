import argparse
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import os


def create_version_dir(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    else:
        while os.path.exists(base_path):
            if base_path[-2] == 'v':
                vers = str(int(base_path[-1]) + 1)
                base_path = base_path[:-1] + vers
            else:
                base_path += '_v1'
        os.makedirs(base_path)

    return base_path


def get_args():
    parser = argparse.ArgumentParser(description='This script loads or trains the CNN.')

    parser.add_argument('--load',
                        default=True,
                        help='True for loading a pretrained model, False otherwise [0]')

    parser.add_argument('--training',
                        default=0,
                        help='True for training, False for evaluation [0]')

    parser.add_argument('--input_width',
                        default=178,
                        help='Width of the input images [178]')

    parser.add_argument('--input_height',
                        default=218,
                        help='Height of the input images [218]')

    parser.add_argument('--crop_width',
                        default=178,
                        help='Width of cropped input images [178]')

    parser.add_argument('--crop_height',
                        default=218,
                        help='Height of cropped input images [178]')

    parser.add_argument('--col_dim',
                        default=3,
                        help='Color Dimensions (3 for RGB, 1 for Grayscale) [3]')

    parser.add_argument('--attribute_count',
                        default=40,
                        help='Number of attributes (labels) [40]')

    # parser.add_argument('--attribute_names',
    #                     '--list',
    #                     nargs='+',
    #                     default="image_name 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Grey_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Side_Burns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young",
    #                     help='Names of attributes to be detected')

    parser.add_argument('--attr_train_size',
                        default=141202,
                        help='Number of samples in training set [141202]')

    # parser.add_argument('--attr_train_size',
    #                     default=141202,
    #                     help='Number of samples in training set [141202]')

    parser.add_argument('--attr_val_size',
                        default=40343,
                        help='Number of samples in validation set [40343]')

    # parser.add_argument('--attr_val_size',
    #                     default=40343,
    #                     help='Number of samples in validation set [40343]')

    parser.add_argument('--attr_test_size',
                        default=20172,
                        help='Number of samples in validation set [20172]')

    # parser.add_argument('--attr_test_size',
    #                     default=20172,
    #                     help='Number of samples in validation set [20172]')

    parser.add_argument('--attr_all_size',
                        default=201717,
                        help='Total Number of samples in the dataset [201717]')

    # parser.add_argument('--attr_all_size',
    #                     default=201717,
    #                     help='Total Number of samples in the dataset [201717]')

    parser.add_argument('--mask_train_size',
                        default=141202,
                        help='Number of samples in training set [5648076]')

    # parser.add_argument('--mask_train_size',
    #                     default=141202,
    #                     help='Number of samples in training set [5648076]')

    parser.add_argument('--mask_val_size',
                        default=40343,
                        help='Number of samples in validation set [1613736]')

    # parser.add_argument('--mask_val_size',
    #                     default=40343,
    #                     help='Number of samples in validation set [1613736]')

    parser.add_argument('--mask_test_size',
                        default=20172,
                        help='Number of samples in validation set [806868]')

    # parser.add_argument('--mask_test_size',
    #                     default=20172,
    #                     help='Number of samples in validation set [806868]')

    parser.add_argument('--mask_all_size',
                        default=201717,
                        help='Total Number of samples in the dataset [8068680]')

    # parser.add_argument('--mask_all_size',
    #                     default=201717,
    #                     help='Total Number of samples in the dataset [8068680]')


    parser.add_argument('--balancing',
                        default=0,
                        help='True for training with batch balancing, False otherwise [0]')

    parser.add_argument('--batch_size',
                        default=12,
                        help='Batch size for images [32]')

    parser.add_argument('--train_epoch',
                        default=22,
                        help='Number of training epochs [22]')

    parser.add_argument('--lr',
                        default=0.001,
                        help='Learning rate [0.001]')

    parser.add_argument('--dropout',
                        default=True,
                        help='True for using dropout, False otherwise [True]')

    parser.add_argument('--dropout_rate',
                        default=0.5,
                        help='Dropout probability [0.5]')

    parser.add_argument('--image_path',
                        default='../../../PycharmProjects/facial_segmentation/landmarked_images/',
                        help='Path to input data [./data/images/]')

    parser.add_argument('--attr_label_path',
                        default='../../../PycharmProjects/facial_segmentation/updated_list_attr_celeba.txt',
                        help='Path to input data labels [~/PycharmProject/facial_segmentation/updated_list_attr_celeba.txt')

    parser.add_argument('--mask_image_path',
                        default='../../../PycharmProjects/facial_segmentation/region_masks/',
                        help='Path to input data [./data/images/]')
                        # default=True,
                        # help='True for using dropout, False otherwise [True]')

    # parser.add_argument('--dropout_rate',
    #                     default=0.5,
    #                     help='Dropout probability [0.5]')

    # parser.add_argument('--image_path',
    #                     default='./../facial_segmentation/landmarked_images/',
    #                     help='Path to input data [./data/images/]')

    # parser.add_argument('--attr_label_path',
    #                     default='./../facial_segmentation/updated_list_attr_celeba.txt',
    #                     help='Path to input data labels [./../facial_segmentation/updated_list_attr_celeba.txt')
    #
    # parser.add_argument('--mask_image_path',
    #                     default='./../facial_segmentation/region_masks/',
    #                     help='Path to input data [./data/images/]')
    #
    parser.add_argument('--mask_label_path',
                        default='../../../PycharmProjects/facial_segmentation/mask_labels.txt',
                        help='Path to input data labels [~/PycharmProjectsfacial_segmentation/mask_labels.txt]')

    parser.add_argument('--load_path',
            default='../../saved_models/attr_training_no_pre/',
                        help='Dir for loading models [../../saved_models/]')

    parser.add_argument('--model_number',
                        default=21,
                        help='Which saved model to load within load_path [24]')

    parser.add_argument('--save_path',
                        default='../../saved_models/attr_training_no_pre/',
                        help='Dir for saving models [./saved_models/]')

    parser.add_argument('--save',
                        default=True,
                        help='True for saving the model, False otherwise [True]')

    parser.add_argument('--gpu',
                        default=1,
                        help='Which gpu to run ops on [1]')

    return parser.parse_args()


def print_args(args):
    print('')
    for arg in sorted(vars(args)):
        print('{0:<18} {1:<}'.format(arg, getattr(args, arg)))
    print('')
