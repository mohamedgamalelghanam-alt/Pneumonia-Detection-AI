import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# دالة تحميل الموديل الذكية
@st.cache_resource
def load_my_model():
    try:
        # المحاولة الأولى: التحميل العادي مع تعطيل الـ Compile
        return tf.keras.models.load_model('my_pneumonia_model.h5', compile=False)
    except Exception:
        # المحاولة الثانية: لو النسخة مختلفة تماماً، بنستخدم الـ Legacy loader
        return tf.keras.layers.TFSMLayer('my_pneumonia_model.h5', call_endpoint='serving_default')

model = load_my_model()

# باقي الكود كما هو...
st.title("تشخيص الالتهاب الرئوي بالذكاء الاصطناعي 🩺")
# ... كمل باقي الكود بتاع الـ uploader والـ predict
