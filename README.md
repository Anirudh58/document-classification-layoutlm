# Document Classification using LayoutLM

## About

This repository can be used as a recipe for fine-tuning a LayoutLM model for document classification. The dataset used for training and evaluation is something specific to my use case. However, the code can be easily modified to work with any dataset.

- Resources used:
    - [LayoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlm)
    - [Repo](https://github.com/lucky-verma/Document-Classification-using-LayoutLM/blob/master/Fine_tuning_LayoutLMForSequenceClassification_on_RVL_CDIP.ipynb)

## Steps to reproduce

### Folder Structure

- Clone/unzip the project. 
- Place the dataset in a folder named `data` in the root directory of the project.
- The final folder structure should look something like this:

```bash
├── data
│   ├── images
│   │   ├── 0
│   │   ├── 2
│   │   ├── 4
│   │   ├── 6
│   │   └── 9
│   └── ocr
│       ├── 0
│       ├── 2
│       ├── 4
│       ├── 6
│       └── 9
├── environment.yml
├── models
│   └── layoutlm-model
│       ├── config.json
│       └── pytorch_model.bin
├── notebooks
│   ├── dataset.ipynb
│   ├── dataset.pdf
│   ├── modeling.ipynb
│   └── modeling.pdf
├── README.md
└── src
    ├── dataset.py
    ├── __pycache__
    │   ├── dataset.cpython-39.pyc
    │   └── utils.cpython-39.pyc
    └── utils.py
```

### Environment Setup

- Create a conda environment with the given yml file as follows:

```bash
conda env create -f environment.yml
```

- Activate the environment:

```bash
conda activate dcl
```

- Note: I had configured my system to use my GPU with the following specifications. Some packages may have to be installed manually depending on your system and OS configuration.
    - OS: Ubuntu 22.04
    - GPU: NVIDIA GeForce RTX 3050
    - CUDA: 11.7
    - NVIDIA Driver: 515
    - PyTorch: 2.0.0+cu117 

- I used tesseract to peform OCR as I needed the text as well as bounding box information from the document images. Follow steps from [here](https://tesseract-ocr.github.io/tessdoc/Installation.html)

### Running the code

- Open the `dataset.ipynb` notebook and run the cells in order.
    - Here, you can visualize a few document images and their corresponding bounding boxes.
    - This ensures that the dataset is correctly loaded and the bounding boxes are correctly extracted.

- Open the `modeling.ipynb` notebook and run the cells in order.
    - You can modify some of the hyperparameters in the notebook to see how they affect the model performance.
    - During training, you should see a training accuracy of around 0.98 and a validation accuracy of around 0.9, after 5 epochs. 
    - The model is then saved in the `models` folder.
    - During testing, the trained models are loaded and evaluated with the testing set. Here, you should see a test accuracy of around 0.94

## Classification Statistics for my use case

- Train Accuracy: 0.982
- Validation Accuracy: 0.9
- Test Accuracy: 0.944
- Average Precision: 0.951
- Average Recall: 0.944
- Average F1-Score: 0.946
