#!/usr/bin/env python3
# predict.py - Flower image classifier prediction script

import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import matplotlib.pyplot as plt

def process_image(image):
    """Processes an image into a NumPy array for model prediction"""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0
    return image.numpy()

def predict(image_path, model, top_k=5):
    """Predicts the top K flower classes for an image"""
    # Load and process image
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    
    # Add batch dimension and predict
    expanded_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(expanded_image)[0]
    
    # Get top K predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    top_classes = [str(i) for i in top_indices]
    
    return top_probs, top_classes

def plot_prediction(image_path, probs, classes, class_names):
    plt.figure(figsize=(10,5))
    # Image plot
    plt.subplot(121)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    # Prediction plot
    plt.subplot(122)
    names = [class_names[c] for c in classes]
    plt.barh(names, probs)
    plt.xlabel('Probability')
    plt.tight_layout()
    plt.savefig('prediction_result.png')  # Save the output
    plt.show()
    
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('model', help='Path to saved Keras model')
    parser.add_argument('--top_k', type=int, default=5, 
                       help='Return top K most likely classes')
    parser.add_argument('--category_names', 
                       help='Path to JSON file mapping labels to flower names')
    args = parser.parse_args()

    # Load the model
    try:
        model = tf.keras.models.load_model(
            args.model,
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load class names if provided
    class_names = None
    if args.category_names:
        try:
            with open(args.category_names, 'r') as f:
                class_names = json.load(f)
        except Exception as e:
            print(f"Error loading category names: {e}")
            return

    # Make prediction
    probs, classes = predict(args.image_path, model, args.top_k)

    # Display results
    print("\nPrediction Results:")
    print("------------------")
    for i, (prob, cls) in enumerate(zip(probs, classes)):
        name = class_names[cls] if class_names else cls
        print(f"{i+1}. {name}: {prob:.4f}")

    # Visualize if class names are available
    if class_names:
        plot_prediction(args.image_path, probs, classes, class_names)

if __name__ == '__main__':
    main()