import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# تحميل الموديل
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('my_pneumonia_model.h5', compile=False)

model = load_my_model()

st.title("تشخيص الالتهاب الرئوي بالذكاء الاصطناعي 🩺")
st.write("ارفع صورة أشعة الصدر لتحليلها")

uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    # معالجة الصورة
    img = image.resize((224, 224))
    img_array = np.array(img.convert('RGB')) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # التوقع
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        st.error(f"النتيجة: PNEUMONIA (مصاب) - الدقة: {prediction[0][0]:.2%}")
    else:
        st.success(f"النتيجة: NORMAL (سليم) - الدقة: {1 - prediction[0][0]:.2%}")
