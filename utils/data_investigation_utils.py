import os
import numpy as np
import torch
import scipy.io
from collections import defaultdict as ddict

from utils.derm7pt_data import Derm7pt_data
#from utils.derm7pt_hybrid import Derm7pt_data as Derm7PtHybrid

#Helper Function, to load Concepts and Labels from different Datasets to check if they are suitable for a concept bottleneck model

def load_derm7pt_metadata(data_dir):
    #using the derm7pt dataset class to load the metadata
    dataset = Derm7pt_data(os.path.join(data_dir, "Derm7pt"))
    dataset.get_Data = False
    #dataset.metadata

    attributes = []
    labels = []
    for _, label, concepts in dataset:
        attributes.append(torch.detach(concepts).numpy())
        labels.append(label)


    X = np.array(attributes)
    y = np.array(labels)
    return X, y


#Todo Work in progress
# def load_derm7pt_metadata_hybrid(data_dir):
#     #using the derm7pt dataset class to load the metadata
#     dataset = Derm7PtHybrid(os.path.join(data_dir, "Derm7pt"))
#     dataset.get_Data = False
#     #dataset.metadata
#
#     attributes = []
#     labels = []
#     for _, label, concepts in dataset:
#         attributes.append(torch.detach(concepts).numpy())
#         labels.append(label)
#
#
#     X = np.array(attributes)
#     y = np.array(labels)
#     return X, y
#     print(X.shape, y.shape)


def load_CUB_200_2011(data_dir):
    #ToDo cite https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/data_processing.py
    #Read CUB_200_2011 attribute labels

    uncertainty_map = {1: {1: 0, 2: 0.5, 3: 0.75, 4:1}, #calibrate main label based on uncertainty label
                            0: {1: 0, 2: 0.5, 3: 0.25, 4: 0}}
    attribute_labels_all = ddict(list) #map from image id to a list of attribute labels
    attribute_certainties_all = ddict(list) #map from image id to a list of attribute certainties
    attribute_uncertain_labels_all = ddict(list) #map from image id to a list of attribute labels calibrated for uncertainty
    class_labels_all = ddict(int)

    with open(os.path.join(data_dir, 'CUB_200_2011/attributes/image_attribute_labels.txt'), 'r') as f:
        for line in f:
            file_idx, attribute_idx, attribute_label, attribute_certainty = line.strip().split()[:4]
            attribute_label = int(attribute_label)
            attribute_certainty = int(attribute_certainty)
            uncertain_label = uncertainty_map[attribute_label][attribute_certainty]
            attribute_labels_all[int(file_idx)].append(attribute_label)
            attribute_uncertain_labels_all[int(file_idx)].append(uncertain_label)
            attribute_certainties_all[int(file_idx)].append(attribute_certainty)

    #print("attribute_labels_all[1]: ", attribute_labels_all[1])
    #print("attribute_uncertain_labels_all[1]: ", attribute_uncertain_labels_all[1])

    with open(os.path.join(data_dir, 'CUB_200_2011/image_class_labels.txt'), 'r') as f:
        for line in f:
            file_idx, class_idx = line.strip().split()
            class_idx = int(class_idx)
            class_labels_all[int(file_idx)] = class_idx

    attributes = attribute_labels_all
    labels = class_labels_all

    X = np.array([attributes[i] for i in range(1, 11788)])
    y = np.array([labels[i] for i in range(1, 11788)])
    return X, y


def load_APascal_VOC(data_dir):
    attributes_list = []
    labels_list = []
    label_map = {}  # maps string labels to integers
    next_label_id = 0

    with open(os.path.join(data_dir, "attribute_data/apascal_train.txt"), "r") as f:
        for line in f:
            parts = line.strip().split()

            # image.jpg label x1 y1 x2 y2 attr1 attr2 ... attr64
            image_id = parts[0]
            label_str = parts[1]
            coords = parts[2:6]  # [x1, y1, x2, y2]
            attributes = list(map(int, parts[6:]))

            # Assign integer to label if not already seen
            if label_str not in label_map:
                label_map[label_str] = next_label_id
                next_label_id += 1

            labels_list.append(label_map[label_str])
            attributes_list.append(attributes)

    X = np.array(attributes_list)
    y = np.array(labels_list)
    return X, y


def load_SUNAttributeDB(data_dir):
    mat = scipy.io.loadmat(os.path.join(data_dir ,'images.mat'))

    labels = np.array([x.item().split("/")[1] for x in np.array(mat['images'].flatten())])

    mat = scipy.io.loadmat(os.path.join(data_dir ,'attributeLabels_continuous.mat'))
    X = mat["labels_cv"]
    y = labels

    return X, y