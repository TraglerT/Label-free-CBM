# Bachelor Thesis on Label-free Concept Bottleneck Models 

This Thesis is based on the paper:
T. Oikarinen, S. Das, L. Nguyen and T.-W. Weng, [*Label-free Concept Bottleneck Models*](https://openreview.net/pdf?id=FlCg47MNvBA), ICLR 2023.

The code base is built also on top of the publicly available code for the paper: https://github.com/Trustworthy-ML-Lab/Label-free-CBM

<img src=data/LF-CBM_overview.jpg alt="Overview" width=655 height=400>

## Setup

1. Install Python and PyTorch (Built on python 3.9.25 and pytorch 2.8).
2. Install dependencies by running `pip install -r requirements.txt`
3. Download pretrained models by running  `bash download_models.sh` (they will be unpacked to `saved_models`)
4. Download and process CUB dataset by running `bash download_cub.sh` 
5. Download ResNet18(Places365) backbone by running `bash download_rn18_places.sh`

Commands and instruction to setup can be found in setup.txt

We do not provide download instructions for ImageNet data, to evaluate using your own copy of ImageNet you must set the correct path in `DATASET_ROOTS["imagenet_train"]` and `DATASET_ROOTS["imagenet_val"]` variables in `data_utils.py`.

## Running the models

### 1. Creating Concept Sets (Optional):
A. Create initial concept set using GPT-3 - `GPT_initial_concepts.ipynb`, do this for all 3 prompt types (can be skipped if using the concept sets we have provided). NOTE: This step costs money and you will have to provide your own `openai.api_key`.

B. Process and filter the concept set by running `Conceptset_processor.ipynb` (Alternatively get ConceptNet concepts by running ConceptNet_conceptset.ipynb)

#### 1.2 Any LLM Concept Set Creation:
A. Create initial concept set using - `LLM_initial_concepts.ipynb`. Prompt and model settings are in the notebook.

B. Extract the concept from the LLM output by running `LLM_concept_extraction.ipynb`.

C. Process and filter the concept set by running `Conceptset_processor.ipynb` -> new concept set will be saved in `data/concept_sets/{model}_filtered_new.txt`.

D. (Optional) Clear the saved clip activations for other concepts -> any file including the clip file needs to be deleted (i.e. any file ending in "ViT-B16.pt").

### 3. Train LF-CBM

A. Train a concept bottleneck model on CIFAR10 by running:

`python train_cbm.py --concept_set data/concept_sets/cifar10_filtered.txt`

B. Train a concept bottleneck model by using the provided script: train_Label-freeCBM.bat. Or train_Label-freeCBM_ALIGN.bat for training a model using the ALIGN VLM.



### 4. Evaluate trained models

Evaluate the trained models by running `evaluate_cbm.ipynb`. This measures model accuracy, creates barplots explaining individual decisions and prints final layer weights which are the basis for creating weight visualizations.

Additional evaluations and reproductions of our model editing experiments are available in the notebooks of `experiments` directory.


![](data/lf_cbm_ind_decision.png)

## Sources

CUB dataset: https://www.vision.caltech.edu/datasets/cub_200_2011/

Sparse final layer training: https://github.com/MadryLab/glm_saga

Explanation bar plots adapted from: https://github.com/slundberg/shap

CLIP: https://github.com/openai/CLIP


