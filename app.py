import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# إعداد واجهة الموقع
st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🩺")

@st.cache_resource
def load_my_model():
    # استخدام safe_mode=False هو الحل السحري لتجاهل الكلمات القديمة في الموديل
    try:
        return tf.keras.models.load_model('my_pneumonia_model.h5', compile=False)
    except Exception:
        # محاولة أخيرة لو النسخة معصلجة تماماً
        return tf.keras.layers.TFSMLayer('my_pneumonia_model.h5', call_endpoint='serving_default')

model = load_my_model()

st.title("تشخيص الالتهاب الرئوي بالذكاء الاصطناعي 🩺")
st.write("ارفع صورة أشعة الصدر (Chest X-ray) لتحليلها فوراً")

uploaded_file = st.file_uploader("اختر صورة الأشعة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    with st.spinner('جاري التحليل...'):
        # معالجة الصورة بنفس الحجم اللي الموديل متعود عليه
        img = image.resize((224, 224))
        img_array = np.array(img.convert('RGB')) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # التوقع
        prediction = model.predict(img_array)
        
        st.markdown("---")
        if prediction[0][0] > 0.5:
            st.error(f"⚠️ النتيجة: احتمال وجود التهاب رئوي (PNEUMONIA) - الدقة: {prediction[0][0]:.2%}")
        else:
            st.success(f"✅ النتيجة: الرئة سليمة (NORMAL) - الدقة: {1 - prediction[0][0]:.2%}")
