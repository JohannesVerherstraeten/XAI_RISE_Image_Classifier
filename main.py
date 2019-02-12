"""
This script feeds images from the VOC Pascal 2012 dataset to a convolutional neural network for object detection.
The network is pretrained on the ImageNet ILSVRC dataset, which contains 1000 different object classes.
For each image in the VOC dataset, the network predicts the object class that is most apparent in the image.

This code is based on:
- https://github.com/pytorch/examples/blob/master/imagenet/main.py
- https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864
"""

import torch
import torch.utils.data
import torch.autograd
import torch.tensor
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.models

import matplotlib.pyplot as plt
import numpy as np
import pprint
import os
import requests
import json
import copy


horizontal_resolution = 10
vertical_resolution = 10

if __name__ == '__main__':

    # ==========
    # == DATA ==
    # ==========
    # Data directories and files. Will be created later on if not yet present.
    voc_dataset_dir = os.path.join(os.getcwd(), "data/VOCdevkit")
    imagenet_index_file = os.path.join(os.getcwd(), "data/imagenet_class_index.json")

    # Download the image dataset if not yet present, and create the data loaders for it.
    # Images must be transformed according to https://pytorch.org/docs/stable/torchvision/models.html
    download_voc_dataset = not os.path.exists(voc_dataset_dir)
    print("Download VOC dataset: {}".format(download_voc_dataset))

    img_normalize_mean = [0.485, 0.456, 0.406]
    img_normalize_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(img_normalize_mean, img_normalize_std)])

    # trainset = torchvision.datasets.VOCDetection(root='./data', year='2012', image_set="train",
    #                                              download=download_voc_dataset, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
    #                                           shuffle=True, num_workers=2)

    testset = torchvision.datasets.VOCDetection(root='./data', year='2012', image_set="val",
                                                download=download_voc_dataset, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)
    # (Actually, only one of these data loaders is used since the network doesn't have to be trained anymore.)

    # Download the ImageNet class index if not yet present. The class index is used to map indices to their
    # corresponding class name.
    download_imagenet_index = not os.path.exists(imagenet_index_file)
    print("Download ImageNet class index: {}".format(download_imagenet_index))
    if download_imagenet_index:
        data = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json')
        with open(imagenet_index_file, "w", encoding="utf-8") as file:
            file.write(data.text)

    class_map = json.load(open(imagenet_index_file))
    # class_map looks as follows:
    #   {"0": ["n01440764", "tench"],
    #    "1": ["n01443537", "goldfish"],
    #    "2": ["n01484850", "great_white_shark"],
    #    ...}


    # ===========
    # == MODEL ==
    # ===========
    # Create the model. All models are pre-trained on ImageNet.
    model_names = sorted(name for name in torchvision.models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(torchvision.models.__dict__[name]))
    print("Following models are available: {}".format(model_names))
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    print("Current model: {}".format(str(model.__class__)))
    print()


    # ===============
    # == UTILITIES ==
    # ===============
    # Opposite of transforms.Normalize
    def unnormalize(img):
        img2 = []
        for channel in range(img.shape[0]):
            img2.append(img[channel] * img_normalize_std[channel] + img_normalize_mean[channel])
        return torch.stack(img2)


    # Function to show an image
    def imshow(img):
        img = unnormalize(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # different shape conventions between matplotlib and pytorch
        plt.show()


    # Function to pretty print a (json) dictionary
    pp = pprint.PrettyPrinter(indent=4)
    def pprintdict(dic):
        pp.pprint(dic)


    # Function to map an index to its corresponding object class name
    def index_to_class_name(index):
        return class_map[str(index)][1]


    def predict_class(imggg, expected_class_index):
        # Send the image though the model to get the object prediction
        predictions = model(torch.autograd.Variable(imggg))
        # predictions:
        # - Tensor of shape (batch_size, nb_of_classes)
        # - The class with the highest value is the predicted class

        if expected_class_index is None:
            # Find the best prediction for each image in the batch
            max_value, max_index = torch.max(predictions, dim=1)

            # Find the predicted class for each image in the batch.
            # predicted_classes = [index_to_class_name(int(max_index[batch])) for batch in range(predictions.shape[0])]
            class_index = int(max_index[0])
            predicted_class = index_to_class_name(class_index)
            print("Prediction: " + predicted_class + " with a certainty of " + str(max_value.item()))

            # make_grid joins a batch of images to one large image.
            # imshow(torchvision.utils.make_grid(imggg))

            return class_index, max_value.item()

        else:
            cert = predictions[0][expected_class_index].item()
            print("Certainty of " + str(cert))

            # make_grid joins a batch of images to one large image.
            # imshow(torchvision.utils.make_grid(imggg))

            return expected_class_index, cert

    def black_out(imgg, x_low, x_high, y_low, y_high):
        for x in range(x_low, x_high+1):
            for y in range(y_low, y_high+1):
                imgg[0][0][x][y] = torch.zeros((1, 1))[0][0]
                imgg[0][1][x][y] = torch.zeros((1, 1))[0][0]
                imgg[0][2][x][y] = torch.zeros((1, 1))[0][0]
        return imgg


    def highlight(imgggg, x_low, x_high, y_low, y_high, grad):
        for x in range(x_low, x_high+1):
            for y in range(y_low, y_high+1):
                imgggg[0][0][x][y] = grad
                imgggg[0][1][x][y] = 0
                imgggg[0][2][x][y] = 0

    # TODO: 'img'-naming!

    # =============================
    # == ACTUAL OBJECT DETECTION ==
    # =============================
    # Press q or close the image window to continue to the next image.
    #
    # The image- and label variables may be batches of images and labels. The batch size is defined in the DataLoader.
    for i, (image, label) in enumerate(testloader):
        print(" --- Image {} ---".format(i))

        if image.shape[2] < 224 or image.shape[3] < 224:
            print(" -> Image too small, skipped.")
            continue

        # Original image
        most_likely_index, certainty = predict_class(image, None)

        result_image = copy.deepcopy(image)

        print("Image shape: {}".format(image.shape))
        for x_block in range(horizontal_resolution):
            for y_block in range(vertical_resolution):
                x_lower = int((image.shape[2]+1) / horizontal_resolution) * x_block
                x_upper = int(((image.shape[2]+1) / horizontal_resolution)) * (x_block+1) - 1
                y_lower = int((image.shape[3] + 1) / vertical_resolution) * y_block
                y_upper = int(((image.shape[3] + 1) / vertical_resolution)) * (y_block+1) - 1

                img = copy.deepcopy(image)

                _, cert = predict_class(black_out(img, x_lower, x_upper, y_lower, y_upper), most_likely_index)

                highlight(result_image, x_lower, x_upper, y_lower, y_upper, certainty - cert)

        imshow(torchvision.utils.make_grid(image))
        imshow(torchvision.utils.make_grid(result_image))
