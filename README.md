# Label-free Concept Bottleneck Models

This is the official repository for our paper [Label-Free Concept Bottleneck Models](https://openreview.net/forum?id=FlCg47MNvBA) published at ICLR 2023. LF-CBM is a new framework to transform any neural networks into an interpretable Concept Bottleneck Model (CBM) without the need for labeled concept data.

**Update 6/5/2023**: We have conducted a crowdsourced user study to measure the interpretability of our method, results are available in Appendix B of [our arxiv paper](https://arxiv.org/abs/2304.06129).

<img src=data/LF-CBM_overview.jpg alt="Overview" width=655 height=400>

## Setup

1. Install Python (3.9) and PyTorch (1.13).
2. Install dependencies by running `pip install -r requirements.txt`
3. Download pretrained models by running  `bash download_models.sh` (they will be unpacked to `saved_models`)
4. Download and process CUB dataset by running `bash download_cub.sh` 
5. Download ResNet18(Places365) backbone by running `bash download_rn18_places.sh`

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

### 2. Train LF-CBM

Train a concept bottleneck model on CIFAR10 by running:

`python train_cbm.py --concept_set data/concept_sets/cifar10_filtered.txt`


### 3. Evaluate trained models

Evaluate the trained models by running `evaluate_cbm.ipynb`. This measures model accuracy, creates barplots explaining individual decisions and prints final layer weights which are the basis for creating weight visualizations.

Additional evaluations and reproductions of our model editing experiments are available in the notebooks of `experiments` directory.

## Results

High Accuracy:

|                   |         |          | Dataset |           |          |
|-------------------|---------|----------|---------|-----------|----------|
| Model             | CIFAR10 | CIFAR100 | CUB200  | Places365 | ImageNet |
| Standard          | 88.80%  | 70.10%   | 76.70%  | 48.56%    | 76.13%   |
| Standard (sparse) | 82.96%  | 58.34%   | **75.96%**  | 38.46%    | **74.35%**   |
| Label-free CBM    | **86.37%** | **65.27%**   | 74.59%  | **43.71%**   | 71.98%   |

For commands to train Label-free CBM and Standard (sparse) models on all 5 datasets, see `training_commands.txt`.

Explainable Decsisions:

![](data/lf_cbm_ind_decision.png)

## Sources

CUB dataset: https://www.vision.caltech.edu/datasets/cub_200_2011/

Sparse final layer training: https://github.com/MadryLab/glm_saga

Explanation bar plots adapted from: https://github.com/slundberg/shap

CLIP: https://github.com/openai/CLIP

## Cite this work
T. Oikarinen, S. Das, L. Nguyen and T.-W. Weng, [*Label-free Concept Bottleneck Models*](https://openreview.net/pdf?id=FlCg47MNvBA), ICLR 2023.

```
@inproceedings{oikarinenlabel,
  title={Label-free Concept Bottleneck Models},
  author={Oikarinen, Tuomas and Das, Subhro and Nguyen, Lam M and Weng, Tsui-Wei},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```


