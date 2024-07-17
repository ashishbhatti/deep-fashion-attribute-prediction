# Deep Fashion Attribute Prediction

**Problem Statement: Predicting Deep Fashion Attributes** \
Predict 3 visual attributes (neck type, sleeve length, pattern) of garments (specifically men’s t-shirts) using Deep Learning.

**Background:** E-commerce websites use attributes to help users categorize garments and to navigate through their catalog using filters.

## Install Dependencies
```
pip install -r requirements.txt
```


## Training

**Run the training script:** 
```
python train.py
```
- The trained model size is 711 MB. Because of this reason, haven't yet uploaded.

The script will train a ResNet50 model on the dataset and save the trained weights to `trained_resnet50.h5`.

## Inference

`inference.py` script will use the trained model to predict attributes for the test images.


1. **Place your test images** in the `test_images/` directory.

2. **Run the inference script:**
   ```
   python inference.py
   ```
   - It will generate an Output.csv file with the predicted attribute values



## Directory Structure
```
deep-fashion-attribute-prediction/
├── workflow.md
├── notebooks/
│   ├── deep-fashion-attr-prediction.ipynb
├── requirements.txt
├── train.py
├── trained_resnet50.h5
├── inference.py
├── test_images/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── Output.csv
└── readme.md
```

## Future Work
- Explore custom CNN model and compare their performance with pre-trained models.
- Implement data augmentation techniques to improve model generalization.
- Deploy the application - Docker, Streamlit. 
- Think of a Generative AI application which uses the model. Explore Fashion GPT.