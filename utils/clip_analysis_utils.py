import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



#get true concepts for all image
def parse_attributes_file(file_path, file_name="image_attribute_labels.txt", use_certainty = False):
    result = {}
    file_path = os.path.join(file_path, file_name)

    i = 0
    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue

            if len(line.strip().split()) != 5:
                image_id_str, attribute_id_str, is_present_str, certainty_id_str, unused_str, time_str = line.strip().split()
                if unused_str != '0':
                    print("Warning: unexpected value in unused field:", unused_str)
            else:
                image_id_str, attribute_id_str, is_present_str, certainty_id_str, time_str = line.strip().split()

            image_id = int(image_id_str)
            attribute_id = int(attribute_id_str)
            is_present = int(is_present_str)
            certainty_id = int(certainty_id_str)

            if use_certainty == True:
                # Map is_present to a more fine grained scale, based on certainty (based on Bahadori and Heckerman 2021)
                if certainty_id == 1:   #not visible
                    is_present = 0.5
                elif certainty_id == 2: #guessing
                    is_present = abs(is_present - 2/6)
                elif certainty_id == 3: #probably
                    is_present = abs(is_present - 1/6)
                else:                   #definitely
                    is_present = is_present

            # Initialize image entry if not exists
            if image_id not in result:
                result[image_id] = {"all_present_concepts": []}

            # Add attribute info
            result[image_id][attribute_id] = [is_present, certainty_id]

            # Track present attributes
            if is_present == 1:
                result[image_id]["all_present_concepts"].append(attribute_id)
            i+=1
    return result


#create list of concepts, based on attributes file
def get_cleaned_concepts(file_path, file_name, used_ground_truth_concepts=True):
    cleaned_concepts = []
    cleaned_not_concepts = []
    grouped_concepts = defaultdict(list)
    with open(os.path.join(file_path, file_name), "r") as f:
        classes = f.read().split("\n")
        for item in classes:
            if len(item.strip()) == 0:
                continue
            if used_ground_truth_concepts:
                #for the original CUB attributes
                part = item.split(" ")[1]
                part = part.replace("_", " ")
                category, value = part.split("::")
                category, value = category.strip(), value.strip()
                not_concept = f"not {category} {value}"
                concept = f"{category} {value}"
            else:
                value = item.strip()
                concept = value
                not_concept = f"not {value}"
                category = "General"

            cleaned_concepts.append(concept)
            cleaned_not_concepts.append(not_concept)
            grouped_concepts[category].append(concept)
    return cleaned_concepts, cleaned_not_concepts, grouped_concepts


#gets the per class predicted probabilities for each concept to be present
def parse_attributes_continuous_file(file_path, file_name="class_attribute_labels_continuous.txt"):
    result = []
    file_path = os.path.join(file_path, file_name)
    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            data = line.strip().split()
            data = np.asarray(data, dtype=float)    #convert to float
            result.append(list(data))
    return result

#plot function for ROC curve and AUC calculation: Comparison Clip prediction vs CUB ground truth
def plot_roc_curve(true_y, y_prob, file_name='', concept_name=''):
    plt.style.use("default")
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    auc_value = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{concept_name}, AUC = {auc_value:.2f}')
    if file_name:
        plt.savefig(os.path.join(os.getcwd(), f"test/ROC_curves/{file_name}.png"))
        plt.close()
    else:
        plt.show()
    return auc_value

#get clip similarity for all images and concepts. Also for duplicate concept set with "not" in front of the concept.
def get_clip_similarity_all(clip_name, concepts, my_similarity:callable, not_concepts='', device="cpu"):
    ### Imports only relevant for this function ###
    import clip
    import torch
    from utils import cbm_utils
    from transformers import AlignProcessor, AlignModel
    from torchvision import datasets
    ###

    words = concepts
    if clip_name == "ALIGN":
        clip_model = AlignModel.from_pretrained("kakaobrain/align-base")
        clip_model.to(device)
        clip_preprocess = AlignProcessor.from_pretrained("kakaobrain/align-base")
        text = ["{}".format(word) for word in words]    #tokenization during inference
        batch_size=64 #lower batch size to avoid OOM
    else:
        clip_model, clip_preprocess = clip.load(clip_name, device=device)
        text = clip.tokenize(["{}".format(word) for word in words]).to(device)
        batch_size=256  #lower batch size to avoid OOM

    data = datasets.ImageFolder("data\CUB_200_2011\images", clip_preprocess)
    image_emb_path = f"test\\test_notebook_{clip_name}_all_images.pt"
    if not os.path.exists(image_emb_path):
        cbm_utils.save_clip_image_features(clip_model, data, image_emb_path, batch_size, device, clip_name=clip_name)
    img_features = torch.load(image_emb_path, map_location="cpu").float()

    cbm_utils.save_clip_text_features(clip_model, text, f"saved_activations/test_notebook_{clip_name}.pt", batch_size, force_recalculate=True, device=device, clip_name=clip_name)
    text_embedding = torch.load(f"saved_activations/test_notebook_{clip_name}.pt", map_location="cpu").float()

    if not_concepts != '':
        words = not_concepts
        if clip_name == "ALIGN":
            text = ["{}".format(word) for word in words]
        else:
            text = clip.tokenize(["{}".format(word) for word in words]).to(device)

        cbm_utils.save_clip_text_features(clip_model, text, f"saved_activations/not_test_notebook_{clip_name}.pt", batch_size, force_recalculate=True, device=device, clip_name=clip_name)
        not_text_embedding = torch.load(f"saved_activations/not_test_notebook_{clip_name}.pt", map_location="cpu").float()
        not_text_embedding /= torch.norm(not_text_embedding, dim=1, keepdim=True)

    #normalize:
    img_features /= torch.norm(img_features, dim=1, keepdim=True)
    text_embedding /= torch.norm(text_embedding, dim=1, keepdim=True)

    #print(img_features.shape, text_embedding.shape, not_text_embedding.shape)
    mult_scores = my_similarity(img_features, text_embedding)
    not_mult_scores = None
    if not_concepts != '': not_mult_scores = my_similarity(img_features, not_text_embedding)

    return mult_scores, not_mult_scores