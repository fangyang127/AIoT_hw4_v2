import streamlit as st
import numpy as np
from PIL import Image
import os
import base64
import io
import urllib.parse
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# 設定
BASE_DIR = 'myna'
MODEL_PATH = 'myna_model.h5'

st.set_page_config(page_title='八哥辨識器 (Streamlit)', layout='centered')
st.title('八哥辨識器')
st.write('請上傳一張八哥照片，或從範例中選擇。')

# CSS: 使按鈕顯示為無框、尺寸較小的樣式（影響全域按鈕）
st.markdown("""
<style>
.stButton>button { border: none !important; background: transparent !important; padding: 2px 6px !important; font-size: 12px !important; }
.stButton>button:hover { background: rgba(0,0,0,0.03) !important; }
.stButton>button:focus { outline: none !important; box-shadow:none !important; }
</style>
""", unsafe_allow_html=True)

# 讀取類別名稱
categories = 'crested_myna,javan_myna,common_myna'.split(',')
labels = '土八哥,白尾八哥,家八哥'.split(',')
N = len(categories)


@st.cache_resource
def load_model():
    """快取模型載入，避免每次重新載入"""
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model, True, None
        except Exception as e:
            return None, False, str(e)
    else:
        return None, False, '找不到模型檔 myna_model.h5'


# 載入模型
model, model_loaded, load_error = load_model()
if model_loaded:
    st.success('✓ 已載入模型：' + MODEL_PATH)
else:
    st.warning('⚠ 載入模型失敗：' + (load_error or '未知錯誤'))


def load_and_preprocess(image: Image.Image):
    img = image.convert('RGB')
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    return arr


def predict(img: Image.Image):
    if not model_loaded or model is None:
        return None
    x = load_and_preprocess(img)
    pred = model.predict(x, verbose=0).flatten()
    return pred


# 準備範例列表
sample_images = []
for cat in categories:
    d = os.path.join(BASE_DIR, cat)
    if os.path.isdir(d):
        for fname in os.listdir(d):
            sample_images.append(os.path.join(d, fname))

# 若 URL 有 query param selected，將其轉成選擇的範例
params = st.experimental_get_query_params()
if 'selected' in params:
    try:
        sel_idx = int(params.get('selected')[0])
        if 0 <= sel_idx < len(sample_images):
            st.session_state['selected_example'] = sample_images[sel_idx]
    except Exception:
        pass

# 兩欄佈局：左欄放上傳與畫廊，右欄顯示輸入影像與預測結果
left_col, right_col = st.columns([1, 1])

with left_col:
    # 初始化 session state，用來儲存畫廊點選的範例路徑
    if 'selected_example' not in st.session_state:
        st.session_state['selected_example'] = '(無)'

    uploaded = st.file_uploader('上傳圖片', type=['png', 'jpg', 'jpeg'])
    use_example = st.selectbox('或選擇範例', ['(無)'] + sample_images, index=0)

# 範例縮圖畫廊（分頁顯示與縮小縮圖）
page_size = 12
if 'gallery_page' not in st.session_state:
    st.session_state['gallery_page'] = 0

total = len(sample_images)
total_pages = (total + page_size - 1) // page_size if total > 0 else 1

pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
with pcol1:
    if st.button('上一頁') and st.session_state['gallery_page'] > 0:
        st.session_state['gallery_page'] -= 1
with pcol2:
    st.markdown(f"### 範例畫廊（第 {st.session_state['gallery_page']+1} / {total_pages} 頁）")
with pcol3:
    if st.button('下一頁') and st.session_state['gallery_page'] < total_pages - 1:
        st.session_state['gallery_page'] += 1

    start = st.session_state['gallery_page'] * page_size
    end = min(start + page_size, total)
    cols = st.columns(4)
    thumb_size = (100, 100)
    for idx_in, i in enumerate(range(start, end)):
        img_path = sample_images[i]
        col = cols[idx_in % 4]
        try:
            thumb = Image.open(img_path).convert('RGB')
            thumb.thumbnail(thumb_size)
            with col:
                st.image(thumb, use_column_width=True)
                btn_key = f'btn_sample_{i}'
                # 使用小圖示按鈕（無框）作為選取
                if st.button('🔘', key=btn_key):
                    st.session_state['selected_example'] = img_path
                    use_example = img_path
        except Exception:
            pass

    # 若畫廊有點選，優先使用 session_state 的選擇
    if st.session_state.get('selected_example') and st.session_state['selected_example'] != '(無)':
        use_example = st.session_state['selected_example']

img_to_classify = None
if uploaded is not None:
    img = Image.open(uploaded)
    img_to_classify = img
elif use_example and use_example != '(無)':
    try:
        img = Image.open(use_example)
        img_to_classify = img
    except Exception as e:
        # 在右欄顯示錯誤
        with right_col:
            st.error('讀取範例失敗：' + str(e))

with right_col:
    if img_to_classify is not None:
        st.image(img_to_classify, caption='輸入影像', use_column_width=True)
        if not model_loaded:
            st.error('模型尚未載入，無法預測。')
        else:
            preds = predict(img_to_classify)
            if preds is None:
                st.error('預測失敗。')
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
