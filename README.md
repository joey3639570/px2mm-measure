
# 📏 物件檢測與尺寸量測系統

本系統為一套使用 OpenCV 與 wxPython 製作的即時影像處理與尺寸量測工具，支援前處理步驟管理、輪廓分析、單位轉換（pixel → mm）、影像點選分析等功能，適用於工業相機視覺量測、簡易瑕疵篩選與快速原型開發。

---

## 🚀 功能總覽

### 🎥 影像來源與預覽
- 啟動本機 webcam 並即時顯示畫面
- 凍結畫面進行分析與點選選取輪廓

### 🧪 影像前處理流程
- 支援常見處理步驟：
  - 灰階轉換
  - 高斯模糊（可調整 Kernel Size）
  - 二值化（支援即時滑桿調整 Threshold）
  - 形態學閉運算（Morphological Close）
  - Canny 邊緣偵測
- 支援動態新增／刪除／排序處理步驟
- 支援參數輸入與儲存（如模糊大小、Canny 閾值）

### 📐 尺寸量測與轉換
- 點選輪廓後輸入真實寬度，建立 pixel-to-mm 換算比
- 量測並即時顯示每個輪廓的寬度、高度（mm）

### 📊 輪廓資訊表格
- 顯示每個符合條件的輪廓資訊：
  - 寬度（mm）
  - 高度（mm）
  - 面積（pixel）
  - 邊界框資訊

---

## 🧩 系統架構

- GUI 框架：`wxPython`
- 影像處理：`OpenCV`、`NumPy`
- 資料顯示：`wx.Grid`、`Matplotlib`

---

## 🛠 安裝說明

### 📦 依賴套件

建議使用 Python 3.8～3.11：

```bash
pip install opencv-python wxPython numpy matplotlib scipy
```

> 若在安裝 `wxPython` 時遇到困難，可參考其[官方安裝指南](https://wxpython.org/pages/downloads/index.html)。

---

## ▶️ 執行方式

```bash
python new_detect.py
```

啟動後將出現視窗，依照以下操作進行：

1. 點擊「Start Webcam」開啟即時預覽。
2. 點擊「Freeze Frame for Selection」凍結畫面並選取物件輪廓。
3. 點擊「Set Reference Width (mm)」輸入真實寬度，建立單位轉換。
4. 點擊「Apply Processing with Size Info」或「Live Measurement」開始量測。

---

## 🧩 自訂功能開發建議

你可以擴充以下模組功能：

| 功能模組          | 建議擴充點                                     |
|-------------------|------------------------------------------------|
| `preprocess_image`| 新增自訂前處理方法，如 Histogram Equalization |
| `ContourInfoPanel`| 顯示更進階統計指標，如 aspect ratio、長寬比     |
| `on_canvas_click` | 支援多選輪廓、距離量測等應用場景               |

---

## 📷 範例畫面（建議加上截圖）

```bash
![Sample Image](https://github.com/joey3639570/px2mm-measure/blob/main/demo.png)
```
