# Associate SDE Deep Learning - Assignment

### Assignment Overview

**Problem Statement:** Predict 3 visual features (neck-type, sleeve-length, pattern) of garments (specifically men’s t-shirts) using Deep Learning.

**Dataset:** `classification-assignment.zip`. Preliminary examination of dataset. 
- An `image` directory with 1783 images of Men's Tshirts. 
- Along with an `attributes.csv` file with attribute-value pair for each image. Total entries are 2238.
- Each attribute is divided into N classes: neck-type (7 classes), sleeve-length (4 classes), pattern (10 classes). The values range from [0:N-1].
- There are missing values for attributes, denoted as `#N/A`. 

### Workflow and Techniques

#### 1. **Data Collection and Preprocessing**
   - **Download and Inspect Data:** Download the zip file from the provided link. Check the images and inspect the CSV to understand the data structure and the distribution of missing values (`#N/A`). Understand the file names, and types.
   - **Data Cleaning:** Handle missing values:
     - Replace `#N/A` with a placeholder (e.g., -1) or use an indicator variable to mark missing values.
     - Impute missing values using techniques like mean/mode imputation or more advanced techniques. Like k nearest neighbors.
   - **Preprocess Data:** Preprocess images is needed for training:
     - **Resize:** Resize the images if specific dimension needed by first layer of neural network. 
     - **Data Normalization:** Normalize image pixel values to the range [0, 1] or [-1, 1] depending on the pre-trained model’s requirements.
     - **Apply augmentations:** (e.g., rotation, zoom, flip) to increase the diversity of your training set.
   - **Train-Validation-Test Split:** Split your data into training, validation and test sets to monitor performance.

#### 2. **Exploratory Data Analysis**
   - **Visualization:** Visualize the data to gain insights, using histograms, scatter plots and correlation matrices.
   - **Distribution of classes:** Understand the distribution of classes.

#### 3. **Model Selection and Preprocessing**
   - **Model Architecture:** A new model or choose a pre-existing CNN model, such as ResNet, Inception, or EfficientNet, and use transfer learning. Given the multi-label nature of the problem, ensure that the final layer can output multiple attributes.
   - **Multi-Label Handling:** Use a sigmoid activation function for the final layer instead of softmax since we are predicting multiple labels per image.
   - **Loss Function:** Use binary cross-entropy loss for each attribute. Sum the losses to get the final loss.
   

#### 4. **Training and hyperparameter selection**
   - **Hyperparameter Selection:** Select values for learning rate, batch size etc, and tune these values using the validation set.
   - **Training Loop:** 
     - Initialize the model with pre-trained weights.
     - Fine-tune the model on the provided dataset. Freeze the base network layers and train only the modified layers initially. Gradually unfreeze more layers and fine-tune the entire network. Use a smaller learning rate for the pre-trained layers and a larger learning rate for the newly added layers.
   - **Handling Imbalance:** Use class weights or oversample minority classes if there is an imbalance in the attribute classes.
   - **Early Stopping and Checkpoints:** Implement early stopping to prevent overfitting and save model checkpoints.
   - **Experimentation:** Experiment with different hyperparameters (learning rate, batch size, etc.) and optimizers

#### 5. **Evaluation**

   - **Metrics:** Define metrics like accuracy, precision, recall, and F1-score for each attribute. (or IoU depending on problem type).
   - **Evaluate:** Evaluate the model on the test set.

#### 6. **Inference**
   - **Inference Script:** Write a script to: 
     - **Load the model:** Load the trained model weight
     - **Make predictions on new images:** Pass test images through the model to get predicted attribute values for each garment. 
     - **Save the predictions in the required format:** Create the Output.csv file with the predicted values

#### 7. **Documentation**
   - **ReadMe File:** Write a clear and concise README file explaining the usage of the code. Include clear instructions on how to set up the environment, train the model, and run the inference script. Document any assumptions or decisions made during the process.
   


### Extra Steps
#### 8. **Deployment and Monitoring**
   - **Deployment:** Deploy your model in a production environment (e.g., using Docker, Kubernetes).
   - **Monitoring:** Set up monitoring to track model performance and detect anomalies.

### Best Practices

- **Version Control:** Use Git for version control. Commit your code regularly and write meaningful commit messages.
- **Code Structure:** Organize your code into modules (e.g., data loading, preprocessing, model definition, training, evaluation, inference).
- **Experiment Tracking:** Use tools like TensorBoard or Weights & Biases to track experiments.
- **Reproducibility:** Ensure your code can be easily reproduced by others. Fix random seeds, provide environment details, and use configuration files for hyperparameters.

### Example Workflow

```python
# Data Preprocessing
def preprocess_data(images_folder, attributes_csv):
    # Implement data loading and preprocessing here
    pass

# Model Definition
def get_model():
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    neck_output = tf.keras.layers.Dense(num_neck_classes, activation='sigmoid', name='neck')(x)
    sleeve_output = tf.keras.layers.Dense(num_sleeve_classes, activation='sigmoid', name='sleeve')(x)
    pattern_output = tf.keras.layers.Dense(num_pattern_classes, activation='sigmoid', name='pattern')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=[neck_output, sleeve_output, pattern_output])
    return model

# Compile Model
model = get_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[early_stopping, checkpoint])

# Inference
def predict(images_folder, model):
    # Implement prediction logic here
    pass
```

### Next Steps

1. **Set Up Environment:** Install necessary libraries and tools (TensorFlow/PyTorch, OpenCV, etc.).
2. **Implement Data Loading and Preprocessing:** Write functions to load and preprocess data.
3. **Define and Compile the Model:** Choose a pre-trained model and adjust it for multi-label classification.
4. **Train the Model:** Train your model, monitoring validation performance.
5. **Write Inference Script:** Develop a script to predict attributes on new images.
6. **Document Everything:** Ensure your ReadMe file is comprehensive and clear.



