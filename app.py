import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Register custom FuzzyLayer
@tf.keras.utils.register_keras_serializable()
class FuzzyLayer(tf.keras.layers.Layer):
    def __init__(self, units=1, **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.centers = self.add_weight(shape=(self.units, input_shape[-1]), initializer='uniform', trainable=True)
        self.sigmas = self.add_weight(shape=(self.units, input_shape[-1]), initializer='ones', trainable=True)

    def call(self, inputs):
        inputs_exp = tf.expand_dims(inputs, 1)
        centers_exp = tf.expand_dims(self.centers, 0)
        sigmas_exp = tf.expand_dims(self.sigmas, 0)
        fuzzy_output = tf.exp(-tf.square(inputs_exp - centers_exp) / (2 * tf.square(sigmas_exp)))
        return tf.reduce_sum(fuzzy_output, axis=-1)

    def get_config(self):
        config = super(FuzzyLayer, self).get_config()
        config.update({'units': self.units})
        return config

# Load models
@st.cache_resource
def load_models():
    mv2 = tf.keras.models.load_model("fuzzy_model_mv2.keras", custom_objects={"FuzzyLayer": FuzzyLayer})
    resnet = tf.keras.models.load_model("fuzzy_model_resnet.keras", custom_objects={"FuzzyLayer": FuzzyLayer})
    effnet = tf.keras.models.load_model("fuzzy_model_effnet.keras", custom_objects={"FuzzyLayer": FuzzyLayer})
    return mv2, resnet, effnet

mv2_model, resnet_model, effnet_model = load_models()

# Grad-CAM utility
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB").resize((224, 224))
    array = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(array, axis=0) / 255.0

def display_gradcam_ui(img_array, img_display, model, last_conv_layer_name, pred_score):
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    heatmap = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_display, 0.6, heatmap_colored, 0.4, 0)
    st.image(superimposed_img, caption=f"Pneumonia Detected ({pred_score*100:.2f}% confidence)", channels="BGR")

# UI
st.title("Pneumonia Detection with Fuzzy CNN Ensemble + Grad-CAM")
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_array = preprocess_image(uploaded_file)
    img_display = cv2.cvtColor(np.array(Image.open(uploaded_file).resize((224, 224))), cv2.COLOR_RGB2BGR)

    # Predict using each model
    mv2_pred = mv2_model.predict(img_array)[0][0]
    resnet_pred = resnet_model.predict(img_array)[0][0]
    effnet_pred = effnet_model.predict(img_array)[0][0]

    # Ensemble prediction
    pred = (3 * mv2_pred + 0.5 * effnet_pred + 1 * resnet_pred) / (3 + 0.5 + 1)
    st.subheader("Model Predictions")
    st.write(f"**MobileNetV2**: {mv2_pred:.4f}")
    st.write(f"**ResNet50**: {resnet_pred:.4f}")
    st.write(f"**EfficientNetB0**: {effnet_pred:.4f}")
    st.write(f"### 📊 Ensemble Prediction: {'Pneumonia' if pred > 0.5 else 'Normal'} ({pred*100:.2f}% confidence)")

    if pred > 0.5:
        st.subheader("Grad-CAM Visualization")
        display_gradcam_ui(img_array, img_display, mv2_model, "out_relu", pred)
    else:
        st.info("No pneumonia detected. No Grad-CAM visualization displayed.")
