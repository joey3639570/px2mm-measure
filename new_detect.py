import wx
import wx.grid as gridlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial.distance import euclidean
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import threading
import time

DEFAULT_BINARY_THRESHOLD = 127
DEFAULT_MIN_AREA = 50
MAXIMUM_AREA = 500

# --- 公用函式區 ---
# OpenCV findContours 傳回格式處理函式
# 處理不同版本的 findContours 回傳值結構
def grab_contours(cnts):
    if len(cnts) == 2:
        return cnts[0]
    elif len(cnts) == 3:
        return cnts[1]
    raise Exception("Contours tuple must have length 2 or 3.")


class ContourInfoPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)

        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 5)

        self.labels = ["Width (mm)", "Height (mm)", "Area (pixels)",
                       "Bounding Rect Area", "Rect (W x H)"]
        for idx, label in enumerate(self.labels):
            self.grid.SetColLabelValue(idx, label)

        self.grid.SetRowLabelSize(0)  # ✅ 隱藏 row label
        #self.grid.SetMargins(0, 0)  # ✅ 移除上下邊界

        # Grid 外面要用 sizer 裝起來並啟用擴展
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.grid, 1, flag=wx.EXPAND)
        self.SetSizer(sizer)

        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.grid.Bind(wx.EVT_SIZE, self.on_resize)

    def on_resize(self, event):
        size = self.grid.GetClientSize()
        num_cols = self.grid.GetNumberCols()
        if num_cols > 0 and size.width > 0:
            padding = 4  # 每欄之間間距的保守估計
            col_width = max((size.width - padding * num_cols) // num_cols, 5)
            for i in range(num_cols):
                self.grid.SetColSize(i, col_width)
        self.grid.ForceRefresh()
        if event:
            event.Skip()

    def update_contours(self, contours_info):
        self.grid.ClearGrid()
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

        self.grid.AppendRows(len(contours_info))
        for row, contour in enumerate(contours_info):
            self.grid.SetCellValue(row, 0, f"{contour['width_mm']:.1f}")
            self.grid.SetCellValue(row, 1, f"{contour['height_mm']:.1f}")
            self.grid.SetCellValue(row, 2, str(contour['area']))
            self.grid.SetCellValue(row, 3, str(contour['bounding_rect_area']))
            self.grid.SetCellValue(row, 4, str(contour['bounding_rect_size']))

        self.Layout()
        self.on_resize(None)  # 手動觸發調整欄寬

# --- 加入前處理步驟的 Dialog ---
class AddStepDialog(wx.Dialog):
    def __init__(self, parent, title="Add Preprocessing Step"):
        super().__init__(parent, title=title, size=(400, 300))
        self.selected_step = None  # 使用者選擇的步驟
        self.parameters = {}       # 該步驟的參數設定

        # 各常用步驟對應的預設參數
        self.default_parameters = {
            'Gaussian Blur': {'Kernel Size': '5'},
            'Binary Threshold': {'Threshold': '127'},
            'Morphological Operations': {'Kernel Size': '5'},
            'Canny Edge Detection': {'Threshold': '50'}
        }

        vbox = wx.BoxSizer(wx.VERTICAL)
        steps = ['Gray Conversion', 'Gaussian Blur', 'Binary Threshold', 'Morphological Operations', 'Canny Edge Detection']

        # 建立下拉式選單讓使用者選擇步驟
        self.step_choice = wx.Choice(self, choices=steps)
        self.step_choice.Bind(wx.EVT_CHOICE, self.on_step_selected)
        vbox.Add(self.step_choice, flag=wx.EXPAND | wx.ALL, border=10)

        # 放置參數輸入欄位的 panel
        self.param_panel = wx.Panel(self)
        self.param_sizer = wx.BoxSizer(wx.VERTICAL)
        self.param_panel.SetSizer(self.param_sizer)
        vbox.Add(self.param_panel, flag=wx.EXPAND | wx.ALL, border=10)

        # 加入底部按鈕：新增與取消
        hbox_buttons = wx.BoxSizer(wx.HORIZONTAL)
        add_button = wx.Button(self, label="Add")
        add_button.Bind(wx.EVT_BUTTON, self.on_add)
        hbox_buttons.Add(add_button, flag=wx.RIGHT, border=10)

        cancel_button = wx.Button(self, label="Cancel")
        cancel_button.Bind(wx.EVT_BUTTON, self.on_cancel)
        hbox_buttons.Add(cancel_button)

        vbox.Add(hbox_buttons, flag=wx.ALIGN_CENTER | wx.TOP, border=10)
        self.SetSizer(vbox)

    # 當選擇不同步驟時，動態生成參數欄位
    def on_step_selected(self, event):
        step = self.step_choice.GetString(self.step_choice.GetSelection())
        self.param_sizer.Clear(True)
        self.parameters = {}

        if step in self.default_parameters:
            for param_name, default_value in self.default_parameters[step].items():
                self.add_param_textbox(param_name, default_value)

        self.param_panel.Layout()
        self.Layout()

    # 加入一個參數的文字輸入框
    def add_param_textbox(self, param_name, default_value):
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(self.param_panel, label=param_name)
        textbox = wx.TextCtrl(self.param_panel, value=default_value)
        hbox.Add(label, flag=wx.RIGHT, border=8)
        hbox.Add(textbox, proportion=1)
        self.param_sizer.Add(hbox, flag=wx.EXPAND | wx.ALL, border=5)
        self.parameters[param_name] = textbox

    # 使用者按下「Add」時，儲存選擇
    def on_add(self, event):
        self.selected_step = self.step_choice.GetString(self.step_choice.GetSelection())
        if self.selected_step:
            self.EndModal(wx.ID_OK)
        else:
            wx.MessageBox('Please select a step.', 'Error', wx.OK | wx.ICON_ERROR)

    # 使用者按下「Cancel」時關閉視窗
    def on_cancel(self, event):
        self.EndModal(wx.ID_CANCEL)

    # 取得目前參數的設定內容（字典格式）
    def get_parameters(self):
        return {name: textbox.GetValue() for name, textbox in self.parameters.items()}




# --- 主視窗類別 ---
class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MyFrame, self).__init__(parent, title=title, size=(1920, 1080))

        # 初始化相關變數
        self.pixel_to_mm_ratio = None  # 畫素與毫米的換算比例
        self.selected_contour = None   # 使用者選擇的輪廓
        self.image = None              # 原始影像
        self.capture = None           # 攝影機物件
        self.streaming = False        # 是否正在串流中
        self.stream_thread = None     # 串流執行緒
        self.binary_threshold = DEFAULT_BINARY_THRESHOLD
        self.min_area = DEFAULT_MIN_AREA

        # 建立 GUI 主面板與排版容器
        self.panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.HORIZONTAL)  # 外層橫向排版容器

        control_panel = wx.BoxSizer(wx.VERTICAL)  # 左側控制元件的縱向排版

        # --- 建立按鈕列 ---
        hbox_buttons = wx.BoxSizer(wx.HORIZONTAL)

        # 啟動攝影機按鈕
        self.start_button = wx.Button(self.panel, label='Start Webcam')
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start_webcam)
        hbox_buttons.Add(self.start_button, flag=wx.EXPAND | wx.ALL, border=10)

        # 凍結畫面按鈕
        self.show_frame_button = wx.Button(self.panel, label='Freeze Frame for Selection')
        self.show_frame_button.Bind(wx.EVT_BUTTON, self.on_freeze_frame)
        hbox_buttons.Add(self.show_frame_button, flag=wx.EXPAND | wx.ALL, border=10)

        # 設定參考寬度按鈕
        self.set_ref_button = wx.Button(self.panel, label='Set Reference Width (mm)')
        self.set_ref_button.Bind(wx.EVT_BUTTON, self.on_set_reference_width)
        hbox_buttons.Add(self.set_ref_button, flag=wx.EXPAND | wx.ALL, border=10)

        # 套用前處理與尺寸標記按鈕
        self.process_mm_button = wx.Button(self.panel, label='Apply Processing with Size Info')
        self.process_mm_button.Bind(wx.EVT_BUTTON, self.on_apply_processing_mm)
        hbox_buttons.Add(self.process_mm_button, flag=wx.EXPAND | wx.ALL, border=10)

        # 即時量測按鈕
        self.live_measure_button = wx.Button(self.panel, label='Live Measurement')
        self.live_measure_button.Bind(wx.EVT_BUTTON, self.on_live_measurement)
        hbox_buttons.Add(self.live_measure_button, flag=wx.EXPAND | wx.ALL, border=10)

        control_panel.Add(hbox_buttons, flag=wx.EXPAND | wx.ALL, border=10)

        # --- 建立參數滑桿列 ---
        hbox_sliders = wx.BoxSizer(wx.HORIZONTAL)

        # 二值化閾值滑桿
        self.binary_label = wx.StaticText(self.panel, label='Binary Threshold: 127')
        hbox_sliders.Add(self.binary_label, flag=wx.LEFT | wx.RIGHT, border=8)
        self.binary_slider = wx.Slider(self.panel, value=127, minValue=0, maxValue=255, style=wx.SL_HORIZONTAL)
        self.binary_slider.Bind(wx.EVT_SLIDER, self.on_slider_update)
        hbox_sliders.Add(self.binary_slider, proportion=1)

        # 最小輪廓面積滑桿
        self.area_label = wx.StaticText(self.panel, label=f'Minimum Area: {self.min_area}')
        hbox_sliders.Add(self.area_label, flag=wx.LEFT | wx.RIGHT, border=8)
        self.area_slider = wx.Slider(self.panel, value=self.min_area, minValue=0, maxValue=MAXIMUM_AREA, style=wx.SL_HORIZONTAL)
        self.area_slider.Bind(wx.EVT_SLIDER, self.on_slider_update)
        hbox_sliders.Add(self.area_slider, proportion=1)

        control_panel.Add(hbox_sliders, flag=wx.EXPAND | wx.ALL, border=10)

        # --- 建立前處理步驟與操作 ---
        self.preprocess_steps = ['Gray Conversion', 'Gaussian Blur', 'Binary Threshold', 'Morphological Operations', 'Canny Edge Detection']
        self.preprocess_parameters = {
            'Gaussian Blur': {'Kernel Size': '5'},
            'Binary Threshold': {'Threshold': str(DEFAULT_BINARY_THRESHOLD)},
            'Morphological Operations': {'Kernel Size': '5'},
            'Canny Edge Detection': {'Threshold': '50'}
        }

        # 步驟清單顯示元件
        self.preprocess_listbox = wx.ListBox(self.panel, choices=self.get_preprocess_display(), style=wx.LB_SINGLE)
        control_panel.Add(self.preprocess_listbox, flag=wx.EXPAND | wx.ALL, border=10)

        # 套用前處理按鈕
        # self.preprocess_button = wx.Button(self.panel, label='Apply Pre-processing')
        # self.preprocess_button.Bind(wx.EVT_BUTTON, self.on_apply_preprocess)
        # control_panel.Add(self.preprocess_button, flag=wx.EXPAND | wx.ALL, border=10)

        # 前處理步驟調整區：新增、刪除、上移、下移
        hbox_add_remove = wx.BoxSizer(wx.HORIZONTAL)
        self.add_step_button = wx.Button(self.panel, label='Add Step')
        self.add_step_button.Bind(wx.EVT_BUTTON, self.on_add_step)
        hbox_add_remove.Add(self.add_step_button, flag=wx.EXPAND | wx.ALL, border=10)

        self.remove_step_button = wx.Button(self.panel, label='Remove Step')
        self.remove_step_button.Bind(wx.EVT_BUTTON, self.on_remove_step)
        hbox_add_remove.Add(self.remove_step_button, flag=wx.EXPAND | wx.ALL, border=10)

        self.move_up_button = wx.Button(self.panel, label='Move Up')
        self.move_up_button.Bind(wx.EVT_BUTTON, self.on_move_up_step)
        hbox_add_remove.Add(self.move_up_button, flag=wx.EXPAND | wx.ALL, border=10)

        self.move_down_button = wx.Button(self.panel, label='Move Down')
        self.move_down_button.Bind(wx.EVT_BUTTON, self.on_move_down_step)
        hbox_add_remove.Add(self.move_down_button, flag=wx.EXPAND | wx.ALL, border=10)

        control_panel.Add(hbox_add_remove, flag=wx.EXPAND | wx.ALL, border=10)

        # 初始化輪廓資訊面板
        self.contour_info_panel = ContourInfoPanel(self.panel)
        control_panel.Add(self.contour_info_panel, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        # 將控制面板加入主橫向排版中（左側）
        vbox.Add(control_panel, flag=wx.EXPAND | wx.ALL, border=10)

        # --- 建立圖像顯示畫布（右側） ---
        self.figure = Figure()
        self.canvas = FigureCanvas(self.panel, -1, self.figure)
        vbox.Add(self.canvas, 1, flag=wx.EXPAND | wx.ALL, border=10)

        # 讓使用者可透過滑鼠點擊畫布來選擇輪廓
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        self.panel.SetSizer(vbox)

        # --- 記錄凍結畫面與輪廓列表 ---
        self.freeze_contours = []
        self.frozen_frame = None

    def update_contour_info(self, contours):
        # 取得每個 contour 的資訊
        contours_info = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            rect_area = w * h
            width_mm = w * self.pixel_to_mm_ratio if self.pixel_to_mm_ratio else w
            height_mm = h * self.pixel_to_mm_ratio if self.pixel_to_mm_ratio else h
            contours_info.append({
                'width_mm': width_mm,
                'height_mm': height_mm,
                'area': int(area),
                'bounding_rect_area': int(rect_area),
                'bounding_rect_size': f'{w} x {h}'
            })

        # 更新面板內容
        self.contour_info_panel.update_contours(contours_info)

    def get_preprocess_display(self):
        return [f'{step} ({", ".join([f"{k}: {v}" for k, v in self.preprocess_parameters.get(step, {}).items()])})'
                for step in self.preprocess_steps]

    # --- 啟動攝影機串流 ---
    def on_start_webcam(self, event):
        if self.streaming:
            wx.MessageBox('請先停止其他串流功能（例如 Live Measurement）', '錯誤', wx.OK | wx.ICON_ERROR)
            return

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            wx.MessageBox('無法開啟攝影機', '錯誤', wx.OK | wx.ICON_ERROR)
            return

        self.streaming = True

        def webcam_loop():
            while self.streaming:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                self.image = frame
                processed = self.preprocess_image(frame)[-1]  # 執行前處理，取最後結果
                contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                display = frame.copy()
                for contour in contours:
                    if cv2.contourArea(contour) < self.min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display, f"{w} x {h} px", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                ax.axis('off')
                self.canvas.draw()
                time.sleep(0.03)

        self.stream_thread = threading.Thread(target=webcam_loop, daemon=True)
        self.stream_thread.start()

    # --- 停止攝影機串流 ---

    def stop_webcam(self, event=None):
        if self.streaming:
            self.streaming = False
            if self.capture:
                self.capture.release()

    # --- 凍結攝影機畫面並擷取輪廓 ---

    def on_freeze_frame(self, event):
        if self.capture is None or not self.capture.isOpened():
            wx.MessageBox('尚未啟動攝影機', '錯誤', wx.OK | wx.ICON_ERROR)
            return

        self.streaming = False  # 停止 webcam thread
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.0)

        ret, frame = self.capture.read()
        if not ret:
            wx.MessageBox('無法從攝影機取得影像', '錯誤', wx.OK | wx.ICON_ERROR)
            return

        self.image = frame.copy()
        processed = self.preprocess_image(frame)[-1]  # 前處理圖像
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{w} x {h} px", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        self.canvas.draw()

        # 保存凍結畫面與輪廓
        self.freeze_contours = contours
        self.frozen_frame = frame.copy()

        self.update_contour_info(contours)



    def on_canvas_click(self, event):
        if event.xdata is None or event.ydata is None or not hasattr(self, 'freeze_contours'):
            return

        click_x, click_y = int(event.xdata), int(event.ydata)
        self.selected_contour = None

        # 找出包含點擊座標的輪廓（以 bounding box 判斷）
        for contour in self.freeze_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x <= click_x <= x + w and y <= click_y <= y + h:
                self.selected_contour = contour
                break

        # 若有選中，畫紅框標示
        if self.selected_contour is not None:
            selected_image = self.frozen_frame.copy()
            x, y, w, h = cv2.boundingRect(self.selected_contour)
            cv2.rectangle(selected_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            self.show_full_image(selected_image)

        # --- 顯示單張圖像在畫布上 ---

    def show_full_image(self, image):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        self.canvas.draw()

    # --- 設定像素與毫米的換算比例 ---
    def on_set_reference_width(self, event):
        if self.selected_contour is None:
            wx.MessageBox('請先點選一個輪廓作為參考', '提示', wx.OK | wx.ICON_INFORMATION)
            return

        x, y, w, h = cv2.boundingRect(self.selected_contour)
        dlg = wx.TextEntryDialog(self, f'該輪廓寬度為 {w} 像素\n請輸入實際寬度（mm）:', '設定參考寬度')
        if dlg.ShowModal() == wx.ID_OK:
            try:
                real_width_mm = float(dlg.GetValue())
                self.pixel_to_mm_ratio = real_width_mm / w
                wx.MessageBox(f'設定成功：1 px = {self.pixel_to_mm_ratio:.3f} mm', '成功',
                              wx.OK | wx.ICON_INFORMATION)
            except:
                wx.MessageBox('輸入格式錯誤，請輸入數字', '錯誤', wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    # --- 套用處理並顯示 mm 單位尺寸 ---
    def on_apply_processing_mm(self, event):
        if self.image is None:
            wx.MessageBox('請先啟動攝影機', '錯誤', wx.OK | wx.ICON_ERROR)
            return

        if self.pixel_to_mm_ratio is None:
            wx.MessageBox('請先設定 pixel to mm 轉換比例', '錯誤', wx.OK | wx.ICON_ERROR)
            return

        processed_images = self.preprocess_image(self.image)
        final_processed_image = processed_images[-1]
        contours, _ = cv2.findContours(final_processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = self.image.copy()

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            width_mm = w * self.pixel_to_mm_ratio
            height_mm = h * self.pixel_to_mm_ratio
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f"{width_mm:.1f}mm x {height_mm:.1f}mm", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        self.show_full_image(result)
        self.update_contour_info(contours)

    # --- 啟動即時尺寸量測模式 ---
    def on_live_measurement(self, event):
        if self.pixel_to_mm_ratio is None:
            wx.MessageBox('請先設定 pixel to mm 轉換比例', '錯誤', wx.OK | wx.ICON_ERROR)
            return

        if self.streaming:
            wx.MessageBox('Live Measurement 已啟動', '錯誤', wx.OK | wx.ICON_ERROR)
            return

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            wx.MessageBox('無法開啟攝影機', '錯誤', wx.OK | wx.ICON_ERROR)
            return

        self.streaming = True

        def live_loop():
            while self.streaming:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                self.image = frame
                processed = self.preprocess_image(frame)[-1]
                contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                display = frame.copy()
                for contour in contours:
                    if cv2.contourArea(contour) < self.min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    width_mm = w * self.pixel_to_mm_ratio
                    height_mm = h * self.pixel_to_mm_ratio
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display, f"{width_mm:.1f}mm x {height_mm:.1f}mm", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                ax.axis('off')
                self.canvas.draw()
                time.sleep(0.03)

        self.stream_thread = threading.Thread(target=live_loop, daemon=True)
        self.stream_thread.start()

    # --- 執行前處理流程 ---
    def preprocess_image(self, image):
        processed_images = []
        processed_images.append(image)
        processed_image = image
        for step in self.preprocess_steps:
            params = self.preprocess_parameters.get(step, {})
            if step == 'Gray Conversion':
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            elif step == 'Gaussian Blur':
                ksize = int(params.get('Kernel Size', '5'))
                processed_image = cv2.GaussianBlur(processed_image, (ksize, ksize), 0)
            elif step == 'Binary Threshold':
                threshold = self.binary_threshold
                _, processed_image = cv2.threshold(processed_image, threshold, 255, cv2.THRESH_BINARY)
            elif step == 'Morphological Operations':
                ksize = int(params.get('Kernel Size', '5'))
                kernel = np.ones((ksize, ksize), np.uint8)
                processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)
            elif step == 'Canny Edge Detection':
                threshold = int(params.get('Threshold', '50'))
                processed_image = cv2.Canny(processed_image, threshold, threshold * 3)
            processed_images.append(processed_image)
        return processed_images

    # --- 套用前處理後顯示多張圖像 ---
    def show_multiple_images(self, images):
        self.figure.clear()
        n = len(images)
        nrows = (n + 1) // 2
        for i, img in enumerate(images):
            ax = self.figure.add_subplot(nrows, 2, i + 1)
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis('off')
        self.canvas.draw()

    '''
    # --- 套用目前選擇的前處理步驟 ---
    def on_apply_preprocess(self, event):
        if self.image is None:
            wx.MessageBox('Please start the webcam first.', 'Error', wx.OK | wx.ICON_ERROR)
            return
        selected_steps = [self.preprocess_steps[i] for i in self.preprocess_listbox.GetSelections()]
        self.preprocess_steps = selected_steps
        processed_images = self.preprocess_image(self.image)
        self.show_multiple_images(processed_images)
    '''

    # --- 滑桿更新參數並即時顯示效果 ---
    def on_slider_update(self, event):
        self.binary_threshold = self.binary_slider.GetValue()
        self.binary_label.SetLabel(f'Binary Threshold: {self.binary_threshold}')
        self.min_area = self.area_slider.GetValue()
        self.area_label.SetLabel(f'Minimum Area: {self.min_area}')

    # --- 新增前處理步驟 ---
    def on_add_step(self, event):
        dialog = AddStepDialog(self)
        if dialog.ShowModal() == wx.ID_OK:
            new_step = dialog.selected_step
            if new_step and new_step not in self.preprocess_steps:
                self.preprocess_steps.append(new_step)
                self.preprocess_parameters[new_step] = dialog.get_parameters()
                self.update_preprocess_listbox()
        dialog.Destroy()

    # --- 移除前處理步驟 ---
    def on_remove_step(self, event):
        selection = self.preprocess_listbox.GetSelection()
        if selection != wx.NOT_FOUND:
            step = self.preprocess_steps.pop(selection)
            if step in self.preprocess_parameters:
                del self.preprocess_parameters[step]
            self.update_preprocess_listbox()

    # --- 上移前處理步驟 ---
    def on_move_up_step(self, event):
        selection = self.preprocess_listbox.GetSelection()
        if selection != wx.NOT_FOUND and selection > 0:
            self.preprocess_steps[selection], self.preprocess_steps[selection - 1] = (
                self.preprocess_steps[selection - 1], self.preprocess_steps[selection])
            self.update_preprocess_listbox()
            self.preprocess_listbox.SetSelection(selection - 1)

    # --- 下移前處理步驟 ---
    def on_move_down_step(self, event):
        selection = self.preprocess_listbox.GetSelection()
        if selection != wx.NOT_FOUND and selection < len(self.preprocess_steps) - 1:
            self.preprocess_steps[selection], self.preprocess_steps[selection + 1] = (
                self.preprocess_steps[selection + 1], self.preprocess_steps[selection])
            self.update_preprocess_listbox()
            self.preprocess_listbox.SetSelection(selection + 1)

    # --- 更新清單顯示與同步參數 ---
    def update_preprocess_listbox(self):
        self.preprocess_listbox.Clear()
        self.preprocess_listbox.AppendItems([
            f'{step} ({", ".join([f"{k}: {v}" for k, v in self.preprocess_parameters.get(step, {}).items()])})'
            for step in self.preprocess_steps
        ])

# --- 主程式入口點 ---
class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame(None, title='物件檢測與尺寸量測系統')
        self.SetTopWindow(frame)
        frame.Show()
        return True

    def OnExit(self):
        if hasattr(self, 'frame') and self.frame.streaming:
            self.frame.stop_webcam()
        return 0

if __name__ == '__main__':
    app = MyApp()
    app.MainLoop()