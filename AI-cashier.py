import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import speech_recognition as sr
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QHBoxLayout, 
                            QVBoxLayout, QWidget, QGroupBox, QScrollArea, 
                            QPushButton, QTextEdit, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from datetime import datetime
import os
import time

# Инициализация моделей
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Загрузка YOLOv8 модели
model = YOLO('yolov8n.pt')

# Инициализация синтезатора речи
engine = pyttsx3.init()
engine.setProperty('rate', 180)
engine.setProperty('volume', 0.9)

# Расширенный словарь цен
prices = {
    'cell phone': 299.9,
    'apple': 0.79,
    'banana': 0.39,
    'bottle': 1.49,
    'cup': 2.49,
    'book': 14.99,
    'mouse': 19.99,
    'keyboard': 29.99,
    'clock': 12.99,
    'laptop': 899.99,
    'chair': 59.99,
    'tv': 499.99,
    'vase': 24.99,
    'scissors': 5.99,
    'umbrella': 15.99,
    'backpack': 39.99,
    'handbag': 49.99,
    'tie': 19.99,
    'suitcase': 79.99,
    'frisbee': 9.99,
    'skis': 199.99,
    'snowboard': 249.99,
    'sports ball': 12.99,
    'kite': 14.99,
    'baseball bat': 29.99,
    'baseball glove': 34.99,
    'skateboard': 89.99,
    'surfboard': 199.99,
    'tennis racket': 49.99,
    'wine glass': 8.99,
    'knife': 7.99,
    'spoon': 2.99,
    'bowl': 6.99,
    'potted plant': 19.99
}

class VoiceThread(QThread):
    detected_speech = pyqtSignal(str)
    
    def run(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            while True:
                try:
                    audio = r.listen(source, timeout=3, phrase_time_limit=3)
                    text = r.recognize_google(audio, language="ru-RU").lower()
                    if "всё" in text or "все" in text or "готово" in text:
                        self.detected_speech.emit("checkout")
                    elif "новый" in text or "ещё" in text or "следующий" in text:
                        self.detected_speech.emit("new_customer")
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass
                except sr.WaitTimeoutError:
                    pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Smart Store Assistant")
        self.setGeometry(100, 100, 1200, 700)
        
        # Главный layout
        main_layout = QHBoxLayout()
        
        # Левая панель - камера и управление
        left_panel = QVBoxLayout()
        
        # Камера
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        left_panel.addWidget(self.camera_label)
        
        # Статус камеры
        self.camera_status = QLabel("Камера активна")
        self.camera_status.setAlignment(Qt.AlignCenter)
        self.camera_status.setStyleSheet("font-size: 14px; color: green;")
        left_panel.addWidget(self.camera_status)
        
        # Кнопки управления
        button_layout = QHBoxLayout()
        
        self.start_voice_btn = QPushButton("Активировать голосовое управление")
        self.start_voice_btn.setStyleSheet("font-size: 14px;")
        self.start_voice_btn.clicked.connect(self.start_voice_recognition)
        
        self.checkout_btn = QPushButton("Завершить покупки")
        self.checkout_btn.setStyleSheet("font-size: 14px;")
        self.checkout_btn.clicked.connect(self.generate_receipt)
        
        self.new_customer_btn = QPushButton("Новый покупатель")
        self.new_customer_btn.setStyleSheet("font-size: 14px;")
        self.new_customer_btn.clicked.connect(self.start_new_customer)
        self.new_customer_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_voice_btn)
        button_layout.addWidget(self.checkout_btn)
        button_layout.addWidget(self.new_customer_btn)
        left_panel.addLayout(button_layout)
        
        # Правая панель - информация
        right_panel = QVBoxLayout()
        
        # Группа для информации о распознавании
        recognition_group = QGroupBox("Информация о покупках")
        recognition_layout = QVBoxLayout()
        
        self.status_label = QLabel("Готов к работе")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        self.current_item_label = QLabel("Текущий товар: -")
        self.current_item_label.setStyleSheet("font-size: 14px;")
        
        self.cart_label = QLabel("Корзина:")
        self.cart_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.cart_display = QTextEdit()
        self.cart_display.setReadOnly(True)
        self.cart_display.setStyleSheet("font-size: 12px;")
        
        self.total_label = QLabel("Итого: $0.00")
        self.total_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        
        recognition_layout.addWidget(self.status_label)
        recognition_layout.addWidget(self.current_item_label)
        recognition_layout.addWidget(self.cart_label)
        recognition_layout.addWidget(self.cart_display)
        recognition_layout.addWidget(self.total_label)
        recognition_group.setLayout(recognition_layout)
        
        right_panel.addWidget(recognition_group)
        
        # Добавляем панели в главный layout
        main_layout.addLayout(left_panel, 60)
        main_layout.addLayout(right_panel, 40)
        
        # Центральный виджет
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Настройки камеры
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Инициализация MediaPipe
        self.pose = mp_pose.Pose(min_detection_confidence=0.7)
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)
        
        # Переменные состояния
        self.last_spoken = None
        self.hand_above_head = False
        self.shopping_cart = []
        self.voice_thread = None
        self.checkout_mode = False
        
        # Таймер для обновления камеры
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
    
    def start_voice_recognition(self):
        if not self.voice_thread or not self.voice_thread.isRunning():
            self.voice_thread = VoiceThread()
            self.voice_thread.detected_speech.connect(self.handle_voice_command)
            self.voice_thread.start()
            self.status_label.setText("Слушаю... Скажите 'Всё' для завершения")
            self.status_label.setStyleSheet("color: blue; font-size: 16px; font-weight: bold;")
            engine.say("Голосовое управление активировано. Скажите 'Всё' для завершения покупок")
            engine.runAndWait()
    
    def handle_voice_command(self, command):
        if command == "checkout" and not self.checkout_mode:
            self.generate_receipt()
            # Запускаем таймер для автоматического сброса через 1 секунду
            QTimer.singleShot(1000, self.start_new_customer)
        elif command == "new_customer" and self.checkout_mode:
            self.start_new_customer()
    
    def update_frame(self):
        if self.checkout_mode:
            return
            
        success, frame = self.cap.read()
        if not success:
            self.camera_status.setText("Ошибка камеры")
            self.camera_status.setStyleSheet("font-size: 14px; color: red;")
            return
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Детекция позы и рук
        pose_results = self.pose.process(image)
        hands_results = self.hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        
        # Координаты головы
        head_top = None
        if pose_results.pose_landmarks:
            nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            head_top = (int(nose.x * w), int(nose.y * h))
        
        # Проверяем поднятые руки
        self.hand_above_head = False
        if hands_results.multi_hand_landmarks and head_top:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                wrist_y = int(wrist.y * h)
                if wrist_y < head_top[1]:
                    self.hand_above_head = True
                    break
        
        # Детекция объектов
        detected_obj = None
        obj_bbox = None
        
        if self.hand_above_head:
            results = model(frame, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > 0.6:
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        
                        if label in prices:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            obj_center = ((x1+x2)//2, (y1+y2)//2)
                            
                            for hand_landmarks in hands_results.multi_hand_landmarks:
                                wrist = hand_landmarks.landmark[0]
                                wrist_pos = (int(wrist.x * w), int(wrist.y * h))
                                
                                distance = np.sqrt((obj_center[0]-wrist_pos[0])**2 + 
                                                  (obj_center[1]-wrist_pos[1])**2)
                                
                                if distance < 150:
                                    detected_obj = label
                                    obj_bbox = (x1, y1, x2, y2)
                                    break
                            
                            if detected_obj:
                                break
        
        # Обработка обнаруженного объекта
        if detected_obj and detected_obj != self.last_spoken:
            price = prices[detected_obj]
            
            # Добавляем товар в корзину
            self.shopping_cart.append((detected_obj, price))
            self.update_cart_display()
            
            # Озвучиваем
            engine.say(f"Добавлено: {detected_obj} - {price:.2f} доллар")
            engine.runAndWait()
            
            self.last_spoken = detected_obj
            self.current_item_label.setText(f"Текущий товар: {detected_obj} (${price:.2f})")
            self.status_label.setText("Товар добавлен в корзину")
            self.status_label.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
        elif not detected_obj:
            self.last_spoken = None
            self.current_item_label.setText("Текущий товар: -")
            if self.hand_above_head:
                self.status_label.setText("Поднята рука, но товар не распознан")
                self.status_label.setStyleSheet("color: orange; font-size: 16px; font-weight: bold;")
            else:
                self.status_label.setText("Готов к работе")
                self.status_label.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
        
        # Визуализация
        if detected_obj and obj_bbox:
            x1, y1, x2, y2 = obj_bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{detected_obj} ${prices[detected_obj]:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Отображение кадра
        qt_image = QImage(image.data, w, h, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_image)
        self.camera_label.setPixmap(pixmap)
    
    def update_cart_display(self):
        cart_text = ""
        total = 0.0
        
        for item, price in self.shopping_cart:
            cart_text += f"{item}: ${price:.2f}\n"
            total += price
        
        self.cart_display.setText(cart_text)
        self.total_label.setText(f"Итого: ${total:.2f}")
    
    def generate_receipt(self):
        if not self.shopping_cart:
            self.status_label.setText("Корзина пуста!")
            self.status_label.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
            engine.say("Корзина пуста, нечего оформлять")
            engine.runAndWait()
            return
        
        # Создаем папку для чеков, если ее нет
        if not os.path.exists("receipts"):
            os.makedirs("receipts")
        
        # Генерируем имя файла с текущей датой и временем
        now = datetime.now()
        receipt_filename = f"receipts/receipt_{now.strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Считаем общую сумму
        total = sum(price for _, price in self.shopping_cart)
        
        # Формируем текст чека
        receipt_text = "=== ЧЕК ===\n"
        receipt_text += f"Дата: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        receipt_text += "Товары:\n"
        
        for item, price in self.shopping_cart:
            receipt_text += f"- {item}: ${price:.2f}\n"
        
        receipt_text += f"\nИтого: ${total:.2f}\n"
        receipt_text += "Спасибо за покупку!"
        
        # Сохраняем чек в файл
        with open(receipt_filename, 'w', encoding='utf-8') as f:
            f.write(receipt_text)
        
        # Показываем чек в интерфейсе
        self.cart_display.setText(receipt_text)
        
        # Озвучиваем
        engine.say(f"Оформлен чек на {total:.2f} доллар. Спасибо за покупку!")
        engine.runAndWait()
        
        # Переходим в режим ожидания нового покупателя
        self.checkout_mode = True
        self.checkout_btn.setEnabled(False)
        self.new_customer_btn.setEnabled(True)
        self.start_voice_btn.setEnabled(False)
        self.status_label.setText("Покупки оформлены! Ожидание нового покупателя")
        self.status_label.setStyleSheet("color: blue; font-size: 16px; font-weight: bold;")
    
    def start_new_customer(self):
        # Сбрасываем состояние для нового покупателя
        self.shopping_cart = []
        self.update_cart_display()
        self.checkout_mode = False
        self.checkout_btn.setEnabled(True)
        self.new_customer_btn.setEnabled(False)
        self.start_voice_btn.setEnabled(True)
        self.status_label.setText("Готов к работе с новым покупателем")
        self.status_label.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
        self.current_item_label.setText("Текущий товар: -")
        
        # Озвучиваем
        engine.say("Готов к работе с новым покупателем")
        engine.runAndWait()
    
    def closeEvent(self, event):
        self.timer.stop()
        if self.voice_thread and self.voice_thread.isRunning():
            self.voice_thread.terminate()
        self.cap.release()
        self.pose.close()
        self.hands.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()