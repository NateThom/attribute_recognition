import utils
import os
import torch
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch.autograd import Variable
import cv2
from skimage import transform
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils as torchvision_utils

class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is matched to output_size. If int, smaller
        of image edges is matched to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        # img = transform.resize(image, (new_h, new_w))
        msk = []
        for index in range(len(masks)):
            msk.append(transform.resize(masks[index], (new_h, new_w)))
        msk = np.asarray(msk)

        return {'image': image, 'attributes': sample['attributes'], 'masks': msk}

    # def __call__(self, sample):
    #     image, masks = sample['image'], sample['masks']
    #
    #     h, w = image.shape[:2]
    #     if isinstance(self.output_size, int):
    #         if h > w:
    #             new_h, new_w = self.output_size * h / w, self.output_size
    #         else:
    #             new_h, new_w = self.output_size, self.output_size * w / h
    #     else:
    #         new_h, new_w = self.output_size
    #     new_h, new_w = int(new_h), int(new_w)
    #
    #     img = transform.resize(image, (new_h, new_w))
    #     msk = []
    #     for index in range(len(masks)):
    #         msk.append(transform.resize(masks[index], (new_h, new_w)))
    #     msk = np.asarray(msk)
    #
    #     return {'image': img, 'attributes': sample['attributes'], 'masks': msk}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        msk = []
        for index in range(len(masks)):
            msk.append(masks[index][top: top + new_h, left: left + new_w])
        msk = np.asarray(msk)

        return {'image': image, 'attributes': sample['attributes'], 'masks': masks}

# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors"""
#
#     def __call__(self, sample):
#         image, attributes, masks = sample['image'], sample['attributes'], sample['masks']
#
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = image.transpose((2, 0, 1))
#
#         # temp1 = torch.from_numpy(image)
#         # temp2 = torch.from_numpy(attributes)
#         # temp3 = torch.from_numpy(masks)
#
#         return {'image': torch.from_numpy(image),
#                 'attributes': torch.from_numpy(attributes),
#                 'masks': torch.from_numpy(masks)}

class AttParseNetDataset(Dataset):
    """AttParseNet dataset."""

    def __init__(self, args, transform=None):
        self.args = args
        self.attr_labels = pd.read_csv(args.attr_label_path, sep=',', skiprows=0, usecols=[n for n in range(1, 41)])
        self.input_filenames = pd.read_csv(args.attr_label_path, sep=',', skiprows=0, usecols=[0])
        self.mask_label_filenames = pd.read_csv(args.mask_label_path, sep=',', usecols=[n for n in range(2, 42)])
        self.transform = transform

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.args.image_path, self.input_filenames.iloc[idx, 0][:-4] + ".npy")
        # image = np.load(img_name)
        img_name = os.path.join(self.args.image_path, self.input_filenames.iloc[idx, 0])
        # image = io.imread(img_name)
        # image = cv2.imread(img_name, 0)
        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        # image = image.transpose((2, 0, 1))
        image = TF.to_tensor(image)

        attributes = self.attr_labels.iloc[idx, ]
        attributes = np.asarray(attributes)
        for index in range(len(attributes)):
            if attributes[index] == -1:
                attributes[index] = 0
        attributes = torch.from_numpy(attributes).float()

        # masks = []
        # for filename in self.mask_label_filenames.iloc[idx, ]:
        #     # masks.append(np.load(os.path.join(self.args.mask_label_path, filename[:-4] + ".npy")))
        #     masks.append(cv2.imread(os.path.join(self.args.mask_image_path, filename), 0))
        # # masks = np.asarray(masks)

        masks = None
        for filename in self.mask_label_filenames.iloc[idx, ]:
            if masks == None:
                masks = TF.to_tensor(cv2.imread(os.path.join(self.args.mask_image_path, filename), 0))
            else:
                temp = TF.to_tensor(cv2.imread(os.path.join(self.args.mask_image_path, filename), 0))
                temp_shape = temp.shape
                # masks = torch.cat((temp, masks))
        # masks = np.asarray(masks)
        # masks = torch.from_numpy(masks)

        sample = {'image': image, 'attributes': attributes, 'masks': masks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class AttParseNet(nn.Module):
    def __init__(self):
        super(AttParseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 75, (7,7))
        self.pool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(75, 200, (3,3))
        self.conv3 = nn.Conv2d(200, 300, (3,3))
        self.conv4 = nn.Conv2d(300, 512, (3,3))
        self.conv5 = nn.Conv2d(512, 512, (3,3))
        self.conv6 = nn.Conv2d(512, 40, (3,3))
        self.fc1 = nn.Linear(40 * 96 * 76, 40)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # print("\tIn Model: input size", x.size(), end=" ")
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x_maps = x
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x, x_maps

def show_batch(args, batch):
    images_batch, attributes_batch, masks_batch = batch['image'], \
                                                  batch['attributes'], \
                                                  batch['masks']

    for masks in masks_batch:
        for image in masks:
            # input_grid = torchvision_utils.make_grid(image.squeeze())
            # plt.imshow(input_grid.numpy().transpose((1, 2, 0)))
            # plt.imshow(input_grid.numpy())
            image = image.numpy()
            plt.imshow(image, cmap="gray")
            plt.show()
            print()
            for i in image:
                for j in i:
                    print(j, end=" ")
                print()
            print(image)

start_time = time.time()
args = utils.get_args()

dataset = AttParseNetDataset(args)
# train_indices, val_indices, test_indices = list(range(args.train_size)), \
#                                            list(range(args.train_size, args.train_size + args.val_size)), \
#                                            list(range(args.train_size + args.val_size, args.all_size))

train_indices, val_indices, test_indices = list(range(1000)), \
                                           list(range(1000)), \
                                           list(range(1000))

if args.shuffle:
#     # np.random.seed(args.random_seed)
    np.random.shuffle(train_indices)

train_set = Subset(dataset, train_indices)
val_set = Subset(dataset, val_indices)
test_set = Subset(dataset, test_indices)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=12)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=12)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=12)

########## BATCH LOADING TIME TEST ##########
# progress_counter = 0
# for batch in val_loader:
#     progress_counter += 1
#     show_batch(args, batch)
#     print(progress_counter)
#     print(f"Total time: {time.time() - start_time}")
#     print()
#############################################

net = AttParseNet()
pytorch_total_params = sum(p.numel() for p in net.parameters())

########## Parallelization ##########
net = nn.DataParallel(net)
#####################################

print(net)
print(f"Total parameters in AttParseNet: {pytorch_total_params}")
# params = list(net.parameters())
# print(len(params))
# for i in range(len(params)):
#     print(params[i].size())
#

criterion1 = nn.BCEWithLogitsLoss()
criterion2 = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

PATH = "./saved_models/attparsenet"

########## TRAINING ##########
min_loss = np.inf

# total_batch_time = 0
# total_to_device_time = 0
# total_prediction_time = 0
# total_calculate_loss_time = 0
# total_derive_and_step_time = 0


for epoch in range(args.train_epoch):
    running_total_loss = 0.0
    running_bce_loss = 0.0
    running_l1_loss = 0.0

    # iteration_index = 0
    for iteration_index, sample_batched in enumerate(train_loader):
        # print("____________________")
        # iteration_index += 1
        iteration_time = time.time()

        inputs, attribute_labels, mask_labels = sample_batched['image'], sample_batched['attributes'], sample_batched['masks']
        # inputs, attribute_labels = sample_batched['image'], sample_batched['attributes']

        # get_batch_time = time.time()
        # total_batch_time += get_batch_time - iteration_time
        # print(f"Total Get Batch Time: {total_batch_time}, Average Get Batch Time: {total_batch_time / (iteration_index +1)}")

        # inputs, attribute_labels = inputs.float(), attribute_labels.float()

        show_batch(args, sample_batched)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        inputs, attribute_labels, mask_labels = inputs.to(device), \
                                                attribute_labels.to(device), \
                                                mask_labels.to(device)
        # inputs, attributes = inputs.to(device), attribute_labels.to(device)

        # to_device_time = time.time()
        # total_to_device_time += to_device_time - get_batch_time
        # print(f"Total To Device Time: {total_to_device_time}, Average To Device Time: {total_to_device_time / (iteration_index+1)}")

        optimizer.zero_grad()
        # net.zero_grad()

        # if epoch is 0 and iteration_index is 0:
        #     print(f"Startup time: {time.time() - start_time}")

        # attribute_preds = net(inputs)
        attribute_preds, mask_preds = net(inputs)

        # prediction_time = time.time()
        # total_prediction_time += prediction_time - to_device_time
        # print(f"Total Prediction Time: {total_prediction_time}, Average Prediction Time: {total_prediction_time / (iteration_index+1)}")

        # loss = criterion1(attribute_preds, attribute_labels)
        loss1 = criterion1(attribute_preds, attribute_labels)
        loss2 = torch.sqrt(criterion2(mask_preds, mask_labels))
        loss = loss1 + torch.sqrt(loss2)

        # calculate_loss_time = time.time()
        # total_calculate_loss_time += calculate_loss_time - prediction_time
        # print(f"Calculate Loss Time: {total_calculate_loss_time}, Average Calculate Loss Time: {total_calculate_loss_time / (iteration_index+1)}")

        loss.backward()
        optimizer.step()

        # derive_and_step_time = time.time()
        # total_derive_and_step_time += derive_and_step_time - calculate_loss_time
        # print(f"Derive and Step Time: {total_derive_and_step_time}, Average Derive and Step Time: {total_derive_and_step_time / (iteration_index+1)}")
        # print("____________________\n")

        running_total_loss += loss.item()
        running_bce_loss += loss1.item()
        running_l1_loss += loss2.item()

        print()

        if iteration_index % 10 == 0:
        # if iteration_index % 1 == 0:
            print(f"[{epoch + 1} {iteration_index}] Total loss: {running_total_loss / (iteration_index+1)} "
                  f"BCE loss: {running_bce_loss / (iteration_index+1)} "
                  f"L2 loss: {running_l1_loss / (iteration_index+1)}")
            print(f"Iteration time: {time.time() - iteration_time}")
            print(f"Total time: {time.time() - start_time}")

    if (running_total_loss/len(train_loader)) < min_loss:
        min_loss = running_total_loss / len(train_loader)
        torch.save(net.state_dict(), PATH + f"_{str(epoch)}_{str(running_total_loss)}")

print("Finished Training!")
##############################

########## VALIDATION ##########
# net = AttParseNet()
# net.load_state_dict(torch.load(PATH + "_21_2.0481244325637817"), strict=False)
#
# print(len(val_loader))
# correct = 0
# total = 0
# progress_counter = 0
# for sample_batched in val_loader:
#     progress_counter += 1
#     print(f"Validation Iteration: {progress_counter}")
#     iteration_time = time.time()
#     inputs, attribute_labels, mask_labels = sample_batched['image'], sample_batched['attributes'], sample_batched[
#         'masks']
#
#     inputs, attribute_labels, mask_labels = inputs.float(), \
#                                             attribute_labels.int(), \
#                                             mask_labels.float()
#
#     optimizer.zero_grad()
#     net.zero_grad()
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net.to(device)
#     inputs, attribute_labels, mask_labels = inputs.to(device), attribute_labels.to(device), mask_labels.to(device)
#
#     attribute_preds, mask_preds = net(inputs)
#     attribute_preds = torch.sigmoid(attribute_preds)
#     attribute_preds = torch.round(attribute_preds)
#
#     attribute_positive_preds = torch.ge(attribute_preds, 1)
#     attribute_negative_preds = torch.logical_not(attribute_positive_preds)
#     attribute_positive_labels = torch.ge(attribute_labels, 1)
#     attribute_negative_labels = torch.logical_not(attribute_positive_labels)
#
#     true_positive = torch.sum((attribute_positive_preds & attribute_positive_labels).int())
#     false_positive = torch.sum((attribute_positive_preds & attribute_negative_labels).int())
#     true_negative = torch.sum((attribute_negative_preds & attribute_negative_labels).int())
#     false_negative = torch.sum((attribute_negative_preds & attribute_positive_labels).int())
#
#     correct += true_positive + true_negative
#     total += attribute_labels.size(0) * 40
#
#     print(f"Correct: {correct}, Total: {total}, Accuracy: {float(correct)/float(total)}")
#     print(f"Iteration time: {time.time() - iteration_time}")
#     print(f"Total time: {time.time() - start_time}")

################################### GRAVEYARD ###################################
# ########## Load Labels ##########
# print("Loading labels for attributes and masks...")
#
# # Open "updated_list_attr_celeba.txt" as pandas dfs. One for labels and the other for filenames
# attr_labels = pd.read_csv(args.attr_label_path, sep=',', skiprows=0, usecols=[n for n in range(1, 41)])
# filenames = pd.read_csv(args.attr_label_path, sep=',', skiprows=0, usecols=[0])
#
# # Open "mask_labels.txt" as pandas dfs. One for labels and the other for filenames
# mask_labels = pd.read_csv(args.mask_label_path, sep=',', usecols=[n for n in range(2, 42)])
#
# # convert dfs to np arrays
# attr_labels = np.asarray(attr_labels)
# mask_labels = np.asarray(mask_labels)
# filenames = np.asarray(filenames).flatten()
#
# # append the path to landmarked images to the filenames
# for i in range(len(filenames)):
#     filenames[i] = args.image_path + filenames[i]
#
# # change all negative labels from -1 to 0
# attr_labels[attr_labels < 1] = 0
#
# # split data into evaluation/training/testing
# attr_val_labels = attr_labels[:args.val_size]
# mask_val_labels = mask_labels[:args.val_size]
# val_filenames = filenames[:args.val_size]
#
# attr_train_labels = attr_labels[args.val_size:args.train_size + args.val_size]
# mask_train_labels = mask_labels[args.val_size:args.train_size + args.val_size]
# train_filenames = filenames[args.val_size:args.train_size + args.val_size]
#
# attr_test_labels = attr_labels[args.train_size + args.val_size:]
# mask_test_labels = mask_labels[args.train_size + args.val_size:]
# test_filenames = filenames[args.train_size + args.val_size:]
#
# print(f"Validation labels: {attr_val_labels.shape}")
# print(f"Training labels: {attr_train_labels.shape}")
# print(f"Testing labels: {attr_test_labels.shape}")
# print(f"All labels: {attr_labels.shape}")
# print("attributes loaded...")
# #################################

# scale = Rescale(100)
# crop = RandomCrop(50)
# composed = transforms.Compose([Rescale(150), RandomCrop(90)])

# fig = plt.figure()
# sample = dataset[65]
# for i, tsfrm in enumerate([scale, crop, composed]):
#     transformed_sample = tsfrm(sample)
#
#     ax = plt.subplot(1, 3, i+1)
#     plt.tight_layout()
#     ax.set_title(type(tsfrm).__name__)
#     plt.imshow(transformed_sample['image'])
#
# plt.show()

# transformed_dataset = AttParseNetDataset(args, transform=transforms.Compose([Rescale(185),
#                                                                              RandomCrop(122),
#                                                                              ToTensor()
#                                                                              ]))

# for i in range(len(transformed_dataset)):
#     sample = transformed_dataset[i]
#
#     print(i, sample['image'].shape, sample['attributes'].shape, sample['masks'].shape)
#
#     if i == 3:
#         break

# dataloader = DataLoader(transformed_dataset, batch_size=14, shuffle=True, num_worke

# for i_batch, sample_batched in enumerate(train_loader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['attributes'].size(),
#           sample_batched['masks'].size())
#     show_batch(args, sample_batched)

# for i_batch, sample_batched in enumerate(val_loader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['attributes'].size(),
#           sample_batched['masks'].size())
#     show_batch(args, sample_batched)

# for i_batch, sample_batched in enumerate(test_loader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['attributes'].size(),
#           sample_batched['masks'].size())
#     show_batch(args, sample_batched)