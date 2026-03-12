import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# تحميل الموديل بنمط التوافق
@st.cache_resource
def load_my_model():
    # compile=False بتخلي الموديل يفتح ككتلة أوزان فقط
    return tf.keras.models.load_model('my_pneumonia_model.h5', compile=False)

model = load_my_model()

st.title("تشخيص الالتهاب الرئوي بالذكاء الاصطناعي 🩺")
uploaded_file = st.file_uploader("ارفع صورة الأشعة (Chest X-ray)", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    # تجهيز الصورة للموديل
    img = image.resize((224, 224))
    img_array = np.array(img.convert('RGB')) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # التوقع
    with st.spinner('جاري التحليل...'):
        prediction = model.predict(img_array)
        prob = float(prediction[0][0])
        
        if prob > 0.5:
            st.error(f"النتيجة: احتمال إصابة (PNEUMONIA) - الثقة: {prob:.2%}")
        else:
            st.success(f"النتيجة: الرئة سليمة (NORMAL) - الثقة: {1-prob:.2%}")
