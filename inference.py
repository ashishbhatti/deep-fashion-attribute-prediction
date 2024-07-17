#!/usr/bin/env python3

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def preprocess_image(img_path, target_size):
    """
    Preprocess image to ensure a certain size,
    normalize pixel values, and convert to numpy array.

    Args:
      img_path: Path to the image file.
      target_size: Target size for the image. w x h

    Returns:
      Preprocessed image as a numpy array.
    """
    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


def predict_attributes(image_paths, model_path):
    """
    Function to get attribute predictions for a batch of images

    Args:
      image_paths: List of image paths
      model_path: Path to the trained model

    Returns:
      A DataFrame containing the predicted attributes for each image
    """
    predictions = []

    # load the pre-trained model for inference
    model = tf.keras.models.load_model('./trained_resnet50.h5')

    for img_path in image_paths:
        img = preprocess_image(img_path, (224, 224))
        preds = model.predict(img)

        neck_pred = np.argmax(preds[0], axis=1)[0]
        sleeve_length_pred = np.argmax(preds[1], axis=1)[0]
        pattern_pred = np.argmax(preds[2], axis=1)[0]

        predictions.append({
            'filename': img_path,
            'neck': neck_pred,
            'sleeve_length': sleeve_length_pred,
            'pattern': pattern_pred
        })

    return pd.DataFrame(predictions)


if __name__ == '__main__':
  """
  Example usage: Inference script

  Inputs:  
  image_directory = './test_images'  # replace with your image directory path
  model_path = './trained_resnet50.h5'  # replace above if needed
  
  Outputs:
  Output.csv: csv file containing the predicted attributes for each image.

  """

  image_directory = './test_images'     # replace with your image directory path
  model_path = './trained_resnet50.h5'  # replace with model path

  # generate paths to images
  image_paths = [os.path.join(image_directory, fname) for fname in os.listdir(image_directory) if fname.endswith('.jpg')]

  predicted_attributes = predict_attributes(image_paths)
  print(predicted_attributes)
  
  # Save the updated DataFrame to a new CSV file
  predict_attributes.to_csv('Output.csv')

