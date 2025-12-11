import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# 設定
BASE_DIR = 'myna'

st.set_page_config(page_title='八哥辨識器 (Streamlit)', layout='centered')
st.title('八哥辨識器')
st.write('請上傳一張八哥照片，或從範例中選擇。')

# 讀取類別名稱
categories = 'crested_myna,javan_myna,common_myna'.split(',')
labels = '土八哥,白尾八哥,家八哥'.split(',')
N = len(categories)

# 嘗試載入模型
MODEL_PATH = 'myna_model.h5'
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success('已載入模型：' + MODEL_PATH)
    except Exception as e:
        st.error(f'載入模型失敗：{e}')
else:
    st.warning('找不到模型檔 myna_model.h5，請先執行訓練腳本並產生模型。')


def load_and_preprocess(image: Image.Image):
    img = image.convert('RGB')
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    return arr


def predict(img: Image.Image):
    if model is None:
        return None
    x = load_and_preprocess(img)
    pred = model.predict(x).flatten()
    return pred


# 準備範例列表
sample_images = []
for cat in categories:
    d = os.path.join(BASE_DIR, cat)
    if os.path.isdir(d):
        for fname in os.listdir(d):
            sample_images.append(os.path.join(d, fname))

col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader('上傳圖片', type=['png', 'jpg', 'jpeg'])
    use_example = st.selectbox('或選擇範例', ['(無)'] + sample_images)

with col2:
    st.write('模型輸出')
    placeholder = st.empty()

img_to_classify = None
if uploaded is not None:
    img = Image.open(uploaded)
    img_to_classify = img
elif use_example and use_example != '(無)':
    try:
        img = Image.open(use_example)
        img_to_classify = img
    except Exception as e:
        st.error('讀取範例失敗：' + str(e))

if img_to_classify is not None:
    st.image(img_to_classify, caption='輸入影像', use_column_width=True)
    preds = predict(img_to_classify)
    if preds is None:
        st.error('模型尚未載入，無法預測。')
    else:
        # 顯示前 N 項結果
        top_idx = preds.argsort()[::-1][:N]
        rows = []
        for idx in top_idx:
            rows.append({'label': labels[idx], 'prob': float(preds[idx])})
        # 輸出成表格與長條圖
        st.subheader('預測結果')
        for r in rows:
            st.write(f"{r['label']}: {r['prob']:.4f}")
        st.bar_chart(np.array(preds))

st.sidebar.markdown('---')
st.sidebar.write('設定')
st.sidebar.write('範例數量: ' + str(len(sample_images)))
