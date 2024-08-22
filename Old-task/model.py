import os 
import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
import keras

def dice_coef(y_true, y_pred):
    smooth = 0.005 
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def build_unet(input_size=(128, 128, 2)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    
    # Bridge
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    
    # Decoder
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Concatenate()([conv4, conv2])
    
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Concatenate()([conv5, conv1])
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
    return model


class GesundModel:
    def __init__(self, model_path):
        if model_path is None:
            raise ValueError("Model path is required")
        self.model_path = model_path
        self.model = None

    def load(self):
        self.model = build_unet(input_size=(128, 128, 2))
        self.model.load_weights(self.model_path)
        return self.model

    def preprocess(self, img, batch=False):
        if not batch:
            # Assuming img is a NIFTI file path
            image = sitk.ReadImage(img)
            array = sitk.GetArrayFromImage(image)
            array = np.transpose(array, (2, 1, 0))
            
            # Select middle slice
            middle_slice = array.shape[2] // 2
            original_image = array[:, :, middle_slice]
            
            # Resize and normalize
            image = cv2.resize(original_image, (128, 128))
            image = (image - image.mean()) / image.std()
            image = image.astype("float32")
            
            # Add channel dimension
            image = np.expand_dims(image, axis=-1)
            
            # Duplicate channel to match model input
            image = np.concatenate([image, image], axis=-1)
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image, original_image
        else:
            # Implement batch preprocessing if needed
            pass

    def predict(self, img, batch=False):
        if self.model is None:
            self.load()

        preprocessed_img, original_img = self.preprocess(img, batch)
        prediction = self.model.predict(preprocessed_img)
        return prediction, original_img

    def postprocess(self, pred, batch=False):
        if not batch:
            # Thresholding
            binary_mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Convert to polygons
            polygons = []
            for contour in contours:
                if len(contour) >= 3:
                    polygon = []
                    for point in contour:
                        x, y = point[0]
                        polygon.append({"x": int(x), "y": int(y)})
                    polygons.append({
                        "mask": polygon,
                        "prediction_class": 1,
                        "confidence": float(np.mean(pred[0, :, :, 0][binary_mask == 1]))
                    })
            
            return {"masks": polygons, "status": 200}
        else:
            # Implement batch postprocessing if needed
            pass

    def visualize_prediction(self, img_path, save_path=None):
        prediction, original_img = self.predict(img_path)
        
        # Resize prediction to match original image size
        resized_pred = cv2.resize(prediction[0, :, :, 0], (original_img.shape[1], original_img.shape[0]))
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot original image
        ax1.imshow(original_img, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Plot original image with prediction overlay
        ax2.imshow(original_img, cmap='gray')
        ax2.imshow(resized_pred, cmap='jet', alpha=0.5)
        ax2.set_title('Prediction Overlay')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
