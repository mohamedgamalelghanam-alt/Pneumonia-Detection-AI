import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# دالة تحميل الموديل مع تعطيل الـ Compile لتجنب تعارض النسخ
@st.cache_resource
def load_my_model():
    # استخدام compile=False يحل مشكلة الـ TypeError و Unrecognized keyword
    model = tf.keras.models.load_model('my_pneumonia_model.h5', compile=False)
    return model

model = load_my_model()

st.title("تشخيص الالتهاب الرئوي بالذكاء الاصطناعي 🩺")
st.markdown("---")
st.write("ارفع صورة أشعة الصدر (Chest X-ray) للحصول على تحليل فوري")

uploaded_file = st.file_uploader("اختر صورة الأشعة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة التي تم رفعها', use_container_width=True)
    
    with st.spinner('جاري التحليل...'):
        # معالجة الصورة لتناسب الموديل (224x224)
        img = image.resize((224, 224))
        img_array = np.array(img.convert('RGB')) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # التوقع
        prediction = model.predict(img_array)
        
        st.markdown("### النتيجة:")
        if prediction[0][0] > 0.5:
            st.error(f"⚠️ احتمال وجود التهاب رئوي (PNEUMONIA) - الدقة: {prediction[0][0]:.2%}")
        else:
            st.success(f"✅ الرئة سليمة (NORMAL) - الدقة: {1 - prediction[0][0]:.2%}")
