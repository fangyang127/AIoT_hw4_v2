# 對話紀錄 - 八哥辨識器專案開發

## 專案概述
本紀錄記載了 AIoT hw4_v2（八哥辨識器）專案的開發過程，包括所有功能需求、實現方案及完成情況。

---

## 開發進程

### 1. Data Augmentation 功能新增
**需求**：新增 Data Augmentation 功能
**實現**：
- 新建 `myna/augmentation.py` 模組
- 實作了多種影像擴增函式：
  - `random_flip()`：隨機翻轉（水平/垂直）
  - `random_rotate()`：隨機旋轉（±25度）
  - `random_crop_resize()`：隨機裁切並縮放
  - `random_color()`：隨機調整亮度、對比度、色彩
  - `add_gaussian_noise()`：加入高斯雜訊
  - `random_augment()`：組合 1-3 個隨機擴增操作
- 在主程式 `7114056047_hw4_v2.py` 中加入：
  - `AUGMENT` 布林開關（預設 True）
  - `AUGMENT_COUNT` 參數控制每張影像的擴增數量（預設 2）
  - 在讀圖迴圈中套用擴增

**提交**：Initial commit: add augmentation, streamlit app, README, requirements

---

### 2. 從 Gradio 改為 Streamlit
**需求**：移除 Gradio，改用 Streamlit 建立推論應用
**實現**：
- 新建 `app.py`
- 實作模型載入機制（快取模型以加速啟動）
- 提供上傳圖片與選擇範例功能
- 模型預測與結果顯示

**相關提交**：
- Optimize app: add model caching with @st.cache_resource and .gitignore

---

### 3. 產生依賴清單與文檔
**需求**：建立 `requirements.txt` 和 `README.md`
**實現**：
- `requirements.txt`：列出 tensorflow, streamlit, Pillow, numpy, pandas, matplotlib, gradio
- `README.md`：包含安裝、訓練、執行、Data Augmentation 說明

**提交**：
- 產生 requirements.txt
- 新增 README

---

### 4. 上傳到 GitHub
**需求**：把專案推到 https://github.com/fangyang127/AIoT_hw4_v2.git
**實現**：
- 初始化 git 倉庫
- Commit 所有檔案
- 推送到遠端
- 後續包含模型檔案的推送

**提交**：
- Initial commit: add augmentation, streamlit app, README, requirements
- Add trained model
- Optimize app: add model caching with @st.cache_resource and .gitignore

---

### 5. 範例圖片預覽功能
**需求**：新增範例圖片預覽與選擇功能
**實現**：
- 建立分頁式縮圖畫廊（每頁 12 張，4 欄佈局）
- 加入「上一頁」「下一頁」按鈕
- 每張縮圖下方加「選擇」按鈕
- 使用 `st.session_state` 儲存使用者選擇

**相關提交**：
- Add sample gallery preview with selectable thumbnails (session_state)
- Gallery: smaller thumbnails and pagination

---

### 6. UI 佈局調整
**需求**：調整左右欄佈局
**實現**：
- 左欄：上傳、範例選單、範例畫廊
- 右欄：顯示輸入影像與預測結果

**相關提交**：
- Layout: move input image and predictions to right column
- UI: move gallery into left column (same as uploader)

---

### 7. 按鈕樣式調整
**需求**：改為黑底白框白字按鈕
**實現**：
- 使用全域 CSS 修改按鈕外觀
- 從無框改為有框（白色邊框）
- 黑底白字配色
- 圓角與 hover/focus 效果

**相關提交**：
- Gallery: small borderless icon buttons (CSS) and fix indentation
- UI: make gallery buttons framed (bordered) and adjust button styles
- Fix button background color: use light gray background and visible text color
- UI: change button style to black background, white border and white text

---

### 8. 預測結果長條圖改進
**需求**：把長條圖改為淺綠色，並改成橫式，後來改回直式並依機率排序
**實現**：
- 新增 `pandas` 與 `altair` 依賴
- 使用 Altair 繪製淺綠色長條圖
- 最終改為直式，依機率由大到小排序
- 若 Altair 失敗自動回退至 `st.bar_chart`

**相關提交**：
- UI: change prediction bar chart to light-green Altair chart
- Add altair/vega_datasets to requirements and make Altair chart horizontal
- UI: make Altair chart vertical and sort categories by probability desc

---

### 9. 畫廊樣式微調
**需求**：
- 把標題改為「範例圖片預覽」，頁數字體改為 14px
- 把「選取」按鈕改成「選擇」

**實現**：
- 分離標題與頁數為兩行
- 縮小並居中顯示頁數
- 更新按鈕文字為「選擇」

**相關提交**：
- UI: separate gallery title and page controls; smaller page-number font
- UI: change gallery title to '範例圖片預覽' and set page-number font to 14px
- Gallery: change thumbnail buttons label to '選擇'

---

### 10. README 文檔完善
**需求**：
- 加入參考資料（https://github.com/yenlung/AI-Demo）
- 調整段落順序（參考資料移到簡介下方）
- 刪除「作者」欄位
- 加入線上 Demo 連結

**實現**：
- 新增「參考資料」段落
- 新增「線上 Demo」段落
- 整理文檔結構

**相關提交**：
- Docs: cite yenlung/AI-Demo in README
- Docs: move References under Introduction and remove Author section
- Docs: add demo site link under Introduction

---

## 最終專案成果

### 檔案清單
```
7114056047_hw4_v2.py          - 訓練與儲存模型的主程式
myna/
  ├── augmentation.py          - Data Augmentation 模組
  ├── crested_myna/            - 土八哥影像資料夾
  ├── javan_myna/              - 白尾八哥影像資料夾
  └── common_myna/             - 家八哥影像資料夾
app.py                         - Streamlit 推論應用
myna_model.h5                  - 訓練後的模型（自動生成）
requirements.txt               - Python 依賴清單
README.md                      - 專案文檔
.gitignore                     - Git 忽略規則
```

### 主要功能
1. **Data Augmentation**：支援多種隨機影像變換
2. **模型訓練**：使用 ResNet50V2 + 自訂分類層
3. **Streamlit UI**：
   - 上傳或選擇範例圖片
   - 分頁式範例畫廊（縮圖預覽）
   - 即時模型推論
   - 淺綠色直式長條圖顯示結果
4. **模型快取**：使用 `@st.cache_resource` 加速應用啟動
5. **GitHub 部署**：完整的遠端倉庫和線上 Demo

### 線上 Demo
- https://aiot-hw4-7114056047.streamlit.app/

### 參考資料
- https://github.com/yenlung/AI-Demo

---

## 開發經驗與學習

### 使用的技術棧
- **深度學習框架**：TensorFlow/Keras
- **預訓練模型**：ResNet50V2
- **Web 框架**：Streamlit
- **資料處理**：NumPy, Pandas, PIL
- **數據視覺化**：Altair
- **版本控制**：Git/GitHub

### 關鍵技術決策
1. **Gradio → Streamlit**：更豐富的 UI 控制與佈局靈活性
2. **Altair 長條圖**：相比 Streamlit 原生圖表，提供更好的排序控制
3. **Session State**：用於跨頁面請求保留使用者選擇
4. **@st.cache_resource**：顯著加速模型載入時間
5. **Data Augmentation**：隨機化操作組合，增強訓練效果

---

## 結論

本專案成功整合了：
- 影像增強技術（Data Augmentation）
- 深度學習模型（ResNet50V2）
- Web 應用框架（Streamlit）
- 版本控制與線上部署（GitHub）

通過迭代式開發和使用者反饋，最終交付了一個功能完整、介面友善的八哥影像分類系統。
