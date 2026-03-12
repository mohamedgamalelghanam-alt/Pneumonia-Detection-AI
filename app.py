import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Pneumonia Detection")

@st.cache_resource
def load_my_model():
    # التحميل بدون compile عشان نهرب من خناقة النسخ
    return tf.keras.models.load_model('my_pneumonia_model.h5', compile=False)

model = load_my_model()

st.title("تشخيص الالتهاب الرئوي 🩺")
uploaded_file = st.file_uploader("ارفع صورة الأشعة", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    
    # تجهيز الصورة
    img = image.resize((224, 224))
    img_array = np.array(img.convert('RGB')) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # التوقع
    pred = model.predict(img_array)[0][0]
    if pred > 0.5:
        st.error(f"احتمال إصابة: {pred:.2%}")
    else:
        st.success(f"سليم تماماً: {1-pred:.2%}")
