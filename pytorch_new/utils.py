import argparse

def get_args():
    parser = argparse.ArgumentParser(description='This script loads or trains the CNN.')

    parser.add_argument('--load',
                        default=True,
                        help='True for loading a pretrained model, False otherwise [0]')

    parser.add_argument('--save',
                        default=False,
                        help='True for saving the model, False otherwise [True]')

    parser.add_argument('--training',
                        default=0,
                        help='True for training, False for evaluation [0]')

    parser.add_argument('--euclidean',
                        default=1,
                        help='True for using euclidean loss, False otherwise [1]')

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

    parser.add_argument('--train_size',
                        default=141202,
                        help='Number of samples in training set [141202]')

    parser.add_argument('--val_size',
                        default=40343,
                        help='Number of samples in validation set [40343]')

    parser.add_argument('--test_size',
                        default=20172,
                        help='Number of samples in validation set [20172]')

    parser.add_argument('--all_size',
                        default=201717,
                        help='Total Number of samples in the dataset [201717]')

    parser.add_argument('--shuffle',
                        default=True,
                        help='Shuffle the order of training samples [True]')

    parser.add_argument('--random_seed',
                        default=64,
                        help='Seed for random number generators [64]')

    parser.add_argument('--batch_size',
                        default=64,
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
                        default='/home/nthom/PycharmProjects/facial_segmentation/landmarked_images/',
                        # default='/home/nthom/PycharmProjects/facial_segmentation/landmarked_images_npy/',
                        help='Path to input data [./data/images/]')

    parser.add_argument('--attr_label_path',
                        default='/home/nthom/PycharmProjects/facial_segmentation/updated_list_attr_celeba.txt',
                        help='Path to input data labels [~/PycharmProject/facial_segmentation/updated_list_attr_celeba.txt')

    parser.add_argument('--attr_list',
                        default=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
                                'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                                'Eyeglasses', 'Goatee', 'Grey_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                                'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                                'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Side_Burns',
                                'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                                'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'],
                        help='List of all 40 attributes')

    parser.add_argument('--mask_image_path',
                        default='/home/nthom/PycharmProjects/facial_segmentation/region_masks_corrected_resized/',
                        # default='/home/nthom/PycharmProjects/facial_segmentation/region_masks_npy/',
                        help='Path to input data [./data/images/]')

    parser.add_argument('--mask_label_path',
                        default='../../../PycharmProjects/facial_segmentation/mask_labels.txt',
                        help='Path to input data labels [~/PycharmProjectsfacial_segmentation/mask_labels.txt]')

    if parser.parse_args().euclidean == 1:
        parser.add_argument('--load_path',
                default='../../saved_models/attr_training/',
                            help='Dir for loading models [../../saved_models/]')
        # parser.add_argument('--load_path',
        #                     default='../../saved_models/mask_pretraining/',
        #                     help='Dir for loading models [../../saved_models/]')
    else:
        parser.add_argument('--load_path',
                            default='../../saved_models/attr_training_no_pre/',
                            help='Dir for loading models [../../saved_models/]')

    parser.add_argument('--model_number',
                        default=21,
                        help='Which saved model to load within load_path [24]')
    if parser.parse_args().euclidean == 1:
        parser.add_argument('--save_path',
                            default='../../saved_models/attr_training/',
                            help='Dir for saving models [./saved_models/]')
    else:
        parser.add_argument('--save_path',
                            default='../../saved_models/attr_training_no_pre/',
                            help='Dir for saving models [./saved_models/]')
    #
    # parser.add_argument('--save_path',
    #                     default='../../saved_models/mask_pretraining/',
    #                     help='Dir for saving models [./saved_models/]')

    parser.add_argument('--gpu',
                        default=1,
                        help='Which gpu to run ops on [1]')

    return parser.parse_args()

# Broken because of list of attributes
def print_args(args):
    print('')
    for arg in sorted(vars(args)):
        print('{0:<18} {1:<}'.format(arg, getattr(args, arg)))
    print('')
