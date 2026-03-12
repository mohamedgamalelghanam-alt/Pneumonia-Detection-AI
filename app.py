import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🩺")

@st.cache_resource
def load_my_model():
    try:
        # الحل النهائي: تحميل الموديل كـ 'graph' كامل لتجنب أخطاء الـ InputLayer
        return tf.keras.models.load_model('my_pneumonia_model.h5', compile=False)
    except Exception as e:
        st.error(f"خطأ تقني: {e}")
        return None

model = load_my_model()

st.title("تشخيص الالتهاب الرئوي بالذكاء الاصطناعي 🩺")
st.write("ارفع صورة أشعة الصدر (Chest X-ray) للتحليل")

uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    with st.spinner('جاري الفحص...'):
        # تحضير الصورة (224x224)
        img = image.resize((224, 224))
        img_array = np.array(img.convert('RGB')) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype('float32')
        
        # التوقع باستخدام الـ raw prediction
        prediction = model(img_array, training=False)
        result = prediction.numpy()[0][0]
        
        st.markdown("---")
        if result > 0.5:
            st.error(f"⚠️ النتيجة: احتمال وجود التهاب رئوي (PNEUMONIA) - الدقة: {result:.2%}")
        else:
            st.success(f"✅ النتيجة: الرئة سليمة (NORMAL) - الدقة: {1 - result:.2%}")
