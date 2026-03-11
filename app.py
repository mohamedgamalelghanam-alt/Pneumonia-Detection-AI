import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# إخفاء تحذيرات TensorFlow المزعجة
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺")

@st.cache_resource
def load_my_model():
    # الحل السحري: بنحمل الموديل كـ 'Layer' عشان نهرب من خناقة الـ InputLayer
    try:
        # بنحاول نحمله عادي أولاً مع تجاهل الأخطاء
        model = tf.keras.models.load_model('my_pneumonia_model.h5', compile=False)
        return model
    except Exception:
        # لو فشل، بنستخدم الطريقة البديلة للنسخ القديمة
        st.warning("جاري تشغيل نمط التوافق...")
        return tf.keras.models.load_model('my_pneumonia_model.h5', compile=False, custom_objects=None)

model = load_my_model()

st.title("تشخيص الالتهاب الرئوي بالذكاء الاصطناعي 🩺")
st.write("ارفع صورة أشعة الصدر (Chest X-ray) للتحليل")

uploaded_file = st.file_uploader("اختر صورة الأشعة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    with st.spinner('جاري فحص الصورة...'):
        # المعالجة
        img = image.resize((224, 224))
        img_array = np.array(img.convert('RGB')) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # التوقع
        prediction = model.predict(img_array)
        
        st.markdown("---")
        if prediction[0][0] > 0.5:
            st.error(f"⚠️ النتيجة: احتمال وجود التهاب رئوي (PNEUMONIA)")
            st.info(f"نسبة التأكد: {prediction[0][0]:.2%}")
        else:
            st.success(f"✅ النتيجة: الرئة سليمة (NORMAL)")
            st.info(f"نسبة التأكد: {1 - prediction[0][0]:.2%}")
