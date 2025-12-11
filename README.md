# 八哥辨識器 (AIoT hw4_v2)

簡介
--
這個專案為一個簡單的八哥 (myna) 影像分類範例，包含資料載入、可選的 Data Augmentation、使用 ResNet50V2 進行特徵擷取並加上輸出分類層的訓練腳本；另外提供以 Streamlit 實作的推論介面 `app.py`。

參考資料
--
- 本專案部分功能與介面設計參考： https://github.com/yenlung/AI-Demo

目錄結構（重點）
--
- `7114056047_hw4_v2.py`：訓練與儲存模型的主程式（包含資料讀取與可選 augmentation）。
- `myna/`：資料夾，包含各類別子資料夾與 `augmentation.py`。
- `myna/augmentation.py`：實作隨機擴增函式（flip/rotate/crop/color/noise）。
- `app.py`：Streamlit 應用，提供上傳圖片或選擇範例、模型預測與結果顯示。
- `myna_model.h5`：訓練後儲存的模型（需自行產生或放入）。
- `requirements.txt`：所需套件清單。

安裝
--
建議在虛擬環境中安裝依賴：

```bash
python -m venv .venv
source .venv/bin/activate    # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

訓練模型
--
在專案根目錄下執行訓練腳本：

```bash
python 7114056047_hw4_v2.py
```

訓練完成後會嘗試儲存模型為 `myna_model.h5`（若成功，`app.py` 會自動載入該檔案以執行預測）。

Data Augmentation
--
擴增實作於 `myna/augmentation.py`，主程式中有兩個參數可調：

- `AUGMENT`：設定是否啟用擴增（預設在程式中為 True）。
- `AUGMENT_COUNT`：每張原始影像要產生的擴增數量（整數，預設在程式中可調）。

如需關閉或調整，請編輯 `7114056047_hw4_v2.py` 頂端對應變數。

執行 Streamlit 應用（推論）
--
確保 `myna_model.h5` 已存在於專案根目錄，然後啟動：

```bash
streamlit run app.py
```

在網頁介面你可以：上傳圖片或從範例中選擇現有圖片，系統會顯示每個類別的機率分數與長條圖。

常見問題
--
- 如果 `app.py` 顯示找不到模型，請先執行訓練腳本產生 `myna_model.h5`。
- 若 GPU/TF 相容性有問題，請確認 `tensorflow` 版本與系統相容，或改用 CPU 版本。

下一步建議
--
- 若要把程式部署到 GitHub + Heroku/Streamlit Cloud，可把 `myna_model.h5` 放到 release 或改成程式啟動時從雲端下載。
- 若需更多控制的 augmentation 策略，可在 `myna/augmentation.py` 增加參數化選項並在主程式中暴露給 CLI 或設定檔。
 
