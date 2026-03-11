import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🩺")

@st.cache_resource
def load_my_model():
    try:
        # السر هنا: استخدام compile=False مع التنسيق الجديد
        model = tf.keras.models.load_model('my_pneumonia_model.h5', compile=False)
        return model
    except Exception as e:
        # لو فشل، بنجرب طريقة الـ custom_objects لو الموديل فيه طبقات معينة
        st.error(f"حدث خطأ أثناء تحميل الموديل: {e}")
        return None

model = load_my_model()

st.title("تشخيص الالتهاب الرئوي بالذكاء الاصطناعي 🩺")
st.write("ارفع صورة أشعة الصدر (Chest X-ray) لتحليلها فوراً")

uploaded_file = st.file_uploader("اختر صورة الأشعة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    with st.spinner('جاري التحليل...'):
        # معالجة الصورة بنفس الطريقة اللي الموديل اتدرب عليها
        img = image.resize((224, 224))
        img_array = np.array(img.convert('RGB')) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # التوقع
        prediction = model.predict(img_array)
        
        st.markdown("### النتيجة:")
        # لو النتيجة أكبر من 0.5 يبقى احتمال إصابة
        if prediction[0][0] > 0.5:
            st.error(f"⚠️ احتمال وجود التهاب رئوي (PNEUMONIA) - الثقة: {prediction[0][0]:.2%}")
        else:
            st.success(f"✅ الرئة سليمة (NORMAL) - الثقة: {1 - prediction[0][0]:.2%}")
