#!/usr/bin/env python3
# type: ignore
"""
AIScale Produce Recognition POS System
Designed for Essae-Teraoka retail scales
"""

import sys
import time
import json
import os
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (  # type: ignore
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QProgressBar, QGroupBox, QDialog,
    QMessageBox, QApplication, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt  # type: ignore
from PyQt5.QtGui import QImage, QPixmap  # type: ignore
import cv2
import numpy as np

# Constants for retail display
WINDOW_WIDTH = 1366
WINDOW_HEIGHT = 768

# Camera preview size - optimized for display real estate
# Using 16:9 aspect ratio to match modern camera sensors
PREVIEW_WIDTH = 800  # Reduced to accommodate wider right panel
PREVIEW_HEIGHT = 450  # Maintains 16:9 aspect ratio

# Model input size (MobileNetV3 requirement)
MODEL_INPUT_SIZE = 224

# High contrast colors for 6-bit TN panel
COLOR_BG = "#FFFFFF"
COLOR_TEXT = "#000000"
COLOR_ACCENT = "#0066CC"
COLOR_SUCCESS = "#008800"
COLOR_WARNING = "#CC6600"
COLOR_ERROR = "#CC0000"

# Product database (₹/kg)
PRODUCT_DB = {
    "apple": {"name": "Apple", "price": 120.00, "icon": "🍎"},
    "banana": {"name": "Banana", "price": 40.00, "icon": "🍌"},
    "black_grapes": {"name": "Black Grapes", "price": 80.00, "icon": "🍇"},
    "mango": {"name": "Mango", "price": 150.00, "icon": "🥭"},
    "orange": {"name": "Orange", "price": 60.00, "icon": "🍊"},
    "pomegranate": {"name": "Pomegranate", "price": 200.00, "icon": "🌸"}
}

class CameraThread(QThread):
    """Handles camera capture in separate thread"""
    frameReady = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.active = True
        self.camera = None
        
    def run(self):
        # Try to open camera (0 for default)
        # On RK3568, this will be the JSK camera
        try:
            self.camera = cv2.VideoCapture(0)
            # Set camera to capture at higher resolution
            # JSK-S8130-V3.0 is 5MP (2592×1944)
            # We'll use 1920×1080 for good quality without overwhelming the system
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            print(f"Camera initialized at {self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        except Exception:
            print("Camera not available - using mock mode")
            self.camera = None
            
        while self.active:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    # Resize for display preview (maintaining aspect ratio)
                    display_frame = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT), 
                                             interpolation=cv2.INTER_LINEAR)
                    self.frameReady.emit(display_frame)
                    
                    # Store full resolution frame for inference
                    self.full_frame = frame
            else:
                # Mock mode: generate test pattern
                frame = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "CAMERA MOCK MODE", (350, 270), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                # Draw grid pattern
                for i in range(0, PREVIEW_WIDTH, 100):
                    cv2.line(frame, (i, 0), (i, PREVIEW_HEIGHT), (128, 128, 128), 1)
                for i in range(0, PREVIEW_HEIGHT, 100):
                    cv2.line(frame, (0, i), (PREVIEW_WIDTH, i), (128, 128, 128), 1)
                self.frameReady.emit(frame)
                self.full_frame = frame
            
            self.msleep(33)  # ~30 FPS
            
    def stop(self):
        self.active = False
        if self.camera:
            self.camera.release()
        self.wait()

class InferenceThread(QThread):
    """Handles ML inference in separate thread"""
    resultReady = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.frame_queue = []
        self.model_loaded = False
        self.inference_session = None
        self.quiet_mode = True  # Reduce terminal output
        self.last_prediction = None  # Track last prediction to avoid spam
        
    def load_model(self):
        """Load ONNX or RKNN model for inference"""
        import os
        
        # First, try to load RKNN for RK3568 hardware
        try:
            from rknnlite.api import RKNNLite
            self.use_rknn = True
            print("🔧 RKNNLite detected - using RKNN inference")
            
            model_path = "models/mobilenetv3_produce.rknn"
            if os.path.exists(model_path):
                self.rknn_lite = RKNNLite()
                ret = self.rknn_lite.load_rknn(model_path)
                if ret != 0:
                    raise RuntimeError('Failed to load RKNN model')
                    
                ret = self.rknn_lite.init_runtime()
                if ret != 0:
                    raise RuntimeError('Failed to init RKNN runtime')
                    
                print(f"✅ RKNN model loaded from {model_path}")
                self.model_loaded = True
            else:
                raise FileNotFoundError(f"RKNN model not found at {model_path}")
                
        except ImportError:
            # Fallback to ONNX for development/testing
            self.use_rknn = False
            print("💻 Using ONNX inference (development mode)")
            
            try:
                import onnxruntime as ort
                
                # Try to load the simplified model first
                model_path = "models/mobilenetv3_produce_simplified.onnx"
                if not os.path.exists(model_path):
                    model_path = "models/mobilenetv3_produce.onnx"
                    
                if os.path.exists(model_path):
                    self.inference_session = ort.InferenceSession(model_path)
                    self.input_name = self.inference_session.get_inputs()[0].name
                    self.output_name = self.inference_session.get_outputs()[0].name
                    print(f"✅ ONNX model loaded from {model_path}")
                    self.model_loaded = True
                else:
                    print("⚠️ Model file not found - using mock mode")
                    print(f"Looking for model at: {os.path.abspath(model_path)}")
                    self.model_loaded = True  # Enable mock mode
                    
            except Exception as e:
                print(f"❌ Model loading error: {e}")
                print("Falling back to mock mode")
                self.model_loaded = True  # Enable mock mode
        
    def process_frame(self, frame):
        """Add frame to processing queue with proper preprocessing"""
        # Crop center square from frame for inference
        h, w = frame.shape[:2]
        
        # Calculate center crop (square) - matching validation preprocessing
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        
        # Crop center square
        cropped = frame[y_start:y_start+size, x_start:x_start+size]
        
        # Resize to 256 first (matching validation transform)
        resized_256 = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # Center crop to 224 (matching validation)
        crop_start = (256 - 224) // 2
        model_input = resized_256[crop_start:crop_start+224, crop_start:crop_start+224]
        
        # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
        model_input_rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
        
        # Normalize (matching your training normalization)
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        model_input_float = model_input_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        model_input_norm = (model_input_float - mean) / std
        
        # Add to queue
        self.frame_queue.append({
            'preprocessed': model_input_norm,
            'original': model_input_rgb  # Keep original for display
        })
        
    def run(self):
        while True:
            if self.frame_queue and self.model_loaded:
                frame_data = self.frame_queue.pop(0)
                
                # Measure inference time
                start_time = time.time()
                
                if hasattr(self, 'use_rknn') and self.use_rknn and hasattr(self, 'rknn_lite'):
                    # Real RKNN inference
                    try:
                        # Prepare input - RKNN expects uint8 HWC format
                        input_data = frame_data['preprocessed']
                        input_data = (input_data * 255).astype(np.uint8)
                        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
                        
                        # Run RKNN inference
                        outputs = self.rknn_lite.inference(inputs=[input_data])
                        predictions = outputs[0][0]
                        
                        # Apply softmax to get probabilities
                        exp_predictions = np.exp(predictions - np.max(predictions))
                        softmax_predictions = exp_predictions / exp_predictions.sum()
                        
                        # Get result
                        classes = ["apple", "banana", "black_grapes", "mango", "orange", "pomegranate"]
                        class_idx = np.argmax(softmax_predictions)
                        confidence = float(softmax_predictions[class_idx])
                        
                        # Print top 3 predictions only if changed significantly
                        if not self.quiet_mode or self.last_prediction != classes[class_idx]:
                            top_3_idx = np.argsort(softmax_predictions)[-3:][::-1]
                            print("\nTop 3 predictions (RKNN):")
                            for idx in top_3_idx:
                                print(f"  {classes[idx]}: {softmax_predictions[idx]:.2%}")
                            self.last_prediction = classes[class_idx]
                        
                        result = {
                            'class': classes[class_idx],
                            'confidence': confidence,
                            'inference_time': time.time() - start_time,
                            'timestamp': datetime.now()
                        }
                    except Exception as e:
                        print(f"RKNN inference error: {e}")
                        # Fallback to mock
                        result = self.mock_inference(start_time)
                        
                elif self.inference_session:
                    # Real ONNX inference
                    try:
                        # Prepare input - ONNX expects NCHW format
                        input_tensor = frame_data['preprocessed'].transpose(2, 0, 1)  # HWC to CHW
                        input_tensor = np.expand_dims(input_tensor, 0)  # Add batch dimension
                        input_tensor = input_tensor.astype(np.float32)
                        
                        # Run inference
                        outputs = self.inference_session.run(
                            [self.output_name],
                            {self.input_name: input_tensor}
                        )
                        predictions = outputs[0][0]
                        
                        # Apply softmax to get probabilities
                        exp_predictions = np.exp(predictions - np.max(predictions))
                        softmax_predictions = exp_predictions / exp_predictions.sum()
                        
                        # Get result
                        classes = ["apple", "banana", "black_grapes", "mango", "orange", "pomegranate"]
                        class_idx = np.argmax(softmax_predictions)
                        confidence = float(softmax_predictions[class_idx])
                        
                        # Print top 3 predictions only if changed significantly
                        if not self.quiet_mode or self.last_prediction != classes[class_idx]:
                            top_3_idx = np.argsort(softmax_predictions)[-3:][::-1]
                            print("\nTop 3 predictions (ONNX):")
                            for idx in top_3_idx:
                                print(f"  {classes[idx]}: {softmax_predictions[idx]:.2%}")
                            self.last_prediction = classes[class_idx]
                        
                        result = {
                            'class': classes[class_idx],
                            'confidence': confidence,
                            'inference_time': time.time() - start_time,
                            'timestamp': datetime.now()
                        }
                    except Exception as e:
                        print(f"ONNX inference error: {e}")
                        # Fallback to mock
                        result = self.mock_inference(start_time)
                else:
                    # Mock inference
                    result = self.mock_inference(start_time)
                
                self.resultReady.emit(result)
                
            self.msleep(100)  # 10 Hz inference rate
            
    def mock_inference(self, start_time):
        """Mock inference fallback"""
        import random
        classes = list(PRODUCT_DB.keys())
        selected_class = random.choice(classes)
        confidence = random.uniform(0.85, 0.99)
        
        return {
            'class': selected_class,
            'confidence': confidence,
            'inference_time': time.time() - start_time,
            'timestamp': datetime.now()
        }

class WeightThread(QThread):
    """Handles weight sensor reading from Essae scale"""
    weightReady = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.active = True
        self.serial_port = None
        self.use_serial = False
        
        # Try to initialize serial connection
        try:
            import serial
            import serial.tools.list_ports
            
            # List available serial ports
            ports = serial.tools.list_ports.comports()
            if ports:
                print("\n📡 Available serial ports:")
                for port in ports:
                    print(f"   - {port.device}: {port.description}")
            
            # Common Essae scale settings
            # Adjust these based on your scale model
            SERIAL_PORT = '/dev/ttyUSB0'  # Linux: /dev/ttyUSB0 or /dev/ttyS0
            # SERIAL_PORT = 'COM3'  # Windows: COM3, COM4, etc.
            BAUD_RATE = 9600  # Common rates: 9600, 4800, 2400
            
            self.serial_port = serial.Serial(
                port=SERIAL_PORT,
                baudrate=BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            self.use_serial = True
            print(f"✅ Scale connected on {SERIAL_PORT} at {BAUD_RATE} baud")
            
        except ImportError:
            print("⚠️  pyserial not installed - using mock weight")
            print("   Install with: pip install pyserial")
        except Exception as e:
            print(f"⚠️  Scale not connected: {e}")
            print("   Using mock weight data")
            
    def parse_weight(self, data):
        """Parse weight from Essae scale data format"""
        # Common Essae formats:
        # Format 1: "ST,GS,+0001.500kg\r\n" (stable, gross, weight)
        # Format 2: "W+0001.500\r\n"
        # Format 3: "+0001.500kg\r\n"
        
        try:
            # Remove whitespace and decode
            if isinstance(data, bytes):
                data = data.decode('ascii').strip()
            
            # Extract numeric weight value
            import re
            # Match patterns like +0001.500, -0000.100, etc.
            match = re.search(r'[+-]?(\d+\.?\d*)', data)
            if match:
                weight = float(match.group())
                return abs(weight)  # Return absolute value
                
        except Exception as e:
            print(f"Weight parsing error: {e}")
            
        return None
        
    def run(self):
        """Read weight from Essae scale via serial"""
        while self.active:
            if self.use_serial and self.serial_port:
                try:
                    # Read data from scale
                    if self.serial_port.in_waiting:
                        data = self.serial_port.readline()
                        weight = self.parse_weight(data)
                        if weight is not None:
                            self.weightReady.emit(weight)
                    else:
                        # Send request command if scale needs polling
                        # Some scales need: self.serial_port.write(b'W\r\n')
                        pass
                        
                except Exception as e:
                    print(f"Serial read error: {e}")
                    # Fall back to mock data
                    import random
                    weight = random.uniform(0.1, 2.5)
                    self.weightReady.emit(weight)
            else:
                # Mock weight for development
                import random
                weight = random.uniform(0.1, 2.5)
                self.weightReady.emit(weight)
                
            self.msleep(200)  # 5 Hz update rate
            
    def stop(self):
        self.active = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("📡 Serial port closed")
        self.wait()

class AIScalePOS(QMainWindow):
    """Main POS application window"""
    
    def __init__(self):
        super().__init__()
        self.current_weight = 0.0
        self.current_product = None
        self.current_confidence = 0.0
        self.manual_mode = False  # Flag to prevent inference after manual selection
        self.init_ui()
        self.init_threads()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("AIScale - Intelligent Produce Recognition")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Set application style for high contrast
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLOR_BG};
            }}
            QLabel {{
                color: {COLOR_TEXT};
                font-size: 16px;
            }}
            QPushButton {{
                background-color: {COLOR_ACCENT};
                color: white;
                border: none;
                padding: 20px;
                font-size: 18px;
                font-weight: bold;
                border-radius: 5px;
            }}
            QPushButton:pressed {{
                background-color: #004499;
            }}
        """)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        
        # Left panel - Camera view
        left_panel = self.create_camera_panel()
        main_layout.addWidget(left_panel, 2)
        
        # Right panel - Product info and controls
        right_panel = self.create_info_panel()
        right_panel.setMinimumWidth(450)  # Increased minimum width
        right_panel.setMaximumWidth(500)  # Set maximum to prevent over-expansion
        main_layout.addWidget(right_panel, 1)
        
    def create_camera_panel(self):
        """Create camera view panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Camera feed
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self.camera_label.setMaximumSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self.camera_label.setScaledContents(True)
        self.camera_label.setStyleSheet("""
            border: 3px solid #000000;
            background-color: #000000;
        """)
        
        # Add overlay to show model input area
        self.camera_container = QWidget()
        camera_layout = QGridLayout()
        self.camera_container.setLayout(camera_layout)
        camera_layout.addWidget(self.camera_label, 0, 0)
        
        # Add instruction label
        self.camera_instruction = QLabel("📷 Center produce in frame")
        self.camera_instruction.setStyleSheet("""
            background-color: rgba(0, 0, 0, 180);
            color: white;
            padding: 10px;
            font-size: 18px;
            border-radius: 5px;
        """)
        self.camera_instruction.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.camera_container, alignment=Qt.AlignCenter)
        layout.addWidget(self.camera_instruction, alignment=Qt.AlignCenter)
        
        # Detection status
        self.status_label = QLabel("Ready - Place item on scale")
        self.status_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: bold;
            padding: 15px;
            background-color: #EEEEEE;
            border-radius: 5px;
            margin: 10px 0;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(60)
        self.status_label.setMaximumHeight(80)
        layout.addWidget(self.status_label)
        
        return panel
        
    def create_info_panel(self):
        """Create product information panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Product display
        self.product_group = QGroupBox("Detected Product")
        self.product_group.setMinimumHeight(250)  # Ensure enough height
        product_layout = QVBoxLayout()
        self.product_group.setLayout(product_layout)
        
        # Product icon and name
        self.product_icon = QLabel("🛒")
        self.product_icon.setStyleSheet("font-size: 72px;")
        self.product_icon.setAlignment(Qt.AlignCenter)
        product_layout.addWidget(self.product_icon)
        
        self.product_name = QLabel("Waiting...")
        self.product_name.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
        """)
        self.product_name.setAlignment(Qt.AlignCenter)
        product_layout.addWidget(self.product_name)
        
        # Confidence meter
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                text-align: center;
                font-size: 16px;
                height: 30px;
            }
        """)
        product_layout.addWidget(self.confidence_bar)
        
        layout.addWidget(self.product_group)
        
        # Weight and price display
        self.weight_group = QGroupBox("Weight & Price")
        self.weight_group.setMinimumHeight(200)  # Ensure enough height
        weight_layout = QGridLayout()
        weight_layout.setSpacing(15)  # Add spacing between rows
        weight_layout.setContentsMargins(15, 20, 15, 15)  # Add margins
        self.weight_group.setLayout(weight_layout)
        
        # Weight
        weight_label_title = QLabel("Weight:")
        weight_label_title.setStyleSheet("font-size: 16px; color: #666666;")
        weight_label_title.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        weight_layout.addWidget(weight_label_title, 0, 0)
        self.weight_label = QLabel("0.000 kg")
        self.weight_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        self.weight_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        weight_layout.addWidget(self.weight_label, 0, 1)
        
        # Price per kg
        rate_label_title = QLabel("Rate:")
        rate_label_title.setStyleSheet("font-size: 16px; color: #666666;")
        rate_label_title.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        weight_layout.addWidget(rate_label_title, 1, 0)
        self.rate_label = QLabel("₹0.00/kg")
        self.rate_label.setStyleSheet("font-size: 24px;")
        self.rate_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        weight_layout.addWidget(self.rate_label, 1, 1)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #DDDDDD; height: 2px;")
        weight_layout.addWidget(separator, 2, 0, 1, 2)
        
        # Total price
        total_label_title = QLabel("Total:")
        total_label_title.setStyleSheet("font-size: 18px; color: #666666; font-weight: bold;")
        total_label_title.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        weight_layout.addWidget(total_label_title, 3, 0)
        self.total_label = QLabel("₹0.00")
        self.total_label.setStyleSheet("""
            font-size: 36px; 
            font-weight: bold;
            color: #008800;
        """)
        self.total_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        weight_layout.addWidget(self.total_label, 3, 1)
        
        # Set column widths
        weight_layout.setColumnMinimumWidth(0, 80)  # Label column
        weight_layout.setColumnStretch(0, 0)
        weight_layout.setColumnStretch(1, 1)
        
        layout.addWidget(self.weight_group)
        
        # Control buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(12)  # Add spacing between buttons
        
        self.confirm_btn = QPushButton("✓ Confirm Transaction")
        self.confirm_btn.setMinimumHeight(50)
        self.confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_SUCCESS};
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 5px;
            }}
            QPushButton:pressed {{
                background-color: #006600;
            }}
        """)
        self.confirm_btn.clicked.connect(self.confirm_transaction)
        button_layout.addWidget(self.confirm_btn)
        
        self.manual_btn = QPushButton("📝 Manual Selection")
        self.manual_btn.setMinimumHeight(45)
        self.manual_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066CC;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:pressed {
                background-color: #004499;
            }
        """)
        self.manual_btn.clicked.connect(self.manual_selection)
        button_layout.addWidget(self.manual_btn)
        
        self.clear_btn = QPushButton("✗ Clear")
        self.clear_btn.setMinimumHeight(45)
        self.clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_ERROR};
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }}
            QPushButton:pressed {{
                background-color: #AA0000;
            }}
        """)
        self.clear_btn.clicked.connect(self.clear_transaction)
        button_layout.addWidget(self.clear_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        return panel
        
    def init_threads(self):
        """Initialize worker threads"""
        # Camera thread
        self.camera_thread = CameraThread()
        self.camera_thread.frameReady.connect(self.update_camera)
        self.camera_thread.start()
        
        # Inference thread
        self.inference_thread = InferenceThread()
        self.inference_thread.resultReady.connect(self.update_inference)
        self.inference_thread.load_model()
        self.inference_thread.start()
        
        # Weight thread
        self.weight_thread = WeightThread()
        self.weight_thread.weightReady.connect(self.update_weight)
        self.weight_thread.start()
        
        # Inference timer (trigger when weight is stable)
        self.inference_timer = QTimer()
        self.inference_timer.timeout.connect(self.trigger_inference)
        self.inference_timer.start(500)  # Check every 500ms
        
    def update_camera(self, frame):
        """Update camera display"""
        # Frame is already resized for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw center square guide (shows model input area)
        h, w = frame_rgb.shape[:2]
        square_size = min(h, w) * 0.8  # 80% of smallest dimension
        x_center = w // 2
        y_center = h // 2
        x1 = int(x_center - square_size // 2)
        y1 = int(y_center - square_size // 2)
        x2 = int(x_center + square_size // 2)
        y2 = int(y_center + square_size // 2)
        
        # Draw guide rectangle (dashed effect)
        color = (255, 255, 0) if self.current_product is None else (0, 255, 0)
        thickness = 2 if self.current_product is None else 3
        
        # Draw corner brackets instead of full rectangle for cleaner look
        bracket_len = 50
        cv2.line(frame_rgb, (x1, y1), (x1 + bracket_len, y1), color, thickness)
        cv2.line(frame_rgb, (x1, y1), (x1, y1 + bracket_len), color, thickness)
        cv2.line(frame_rgb, (x2, y1), (x2 - bracket_len, y1), color, thickness)
        cv2.line(frame_rgb, (x2, y1), (x2, y1 + bracket_len), color, thickness)
        cv2.line(frame_rgb, (x1, y2), (x1 + bracket_len, y2), color, thickness)
        cv2.line(frame_rgb, (x1, y2), (x1, y2 - bracket_len), color, thickness)
        cv2.line(frame_rgb, (x2, y2), (x2 - bracket_len, y2), color, thickness)
        cv2.line(frame_rgb, (x2, y2), (x2, y2 - bracket_len), color, thickness)
        
        # Add detection overlay if product detected
        if self.current_product and self.current_confidence > 0.7:
            # Draw filled detection box
            overlay = frame_rgb.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            frame_rgb = cv2.addWeighted(frame_rgb, 0.8, overlay, 0.2, 0)
            
            # Add product label
            product_info = PRODUCT_DB[self.current_product]
            label = f"{product_info['name']} ({self.current_confidence:.0%})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            label_x = x_center - label_size[0] // 2
            label_y = y1 - 10
            
            # Draw label background
            cv2.rectangle(frame_rgb, 
                         (label_x - 10, label_y - 30), 
                         (label_x + label_size[0] + 10, label_y + 5),
                         (0, 255, 0), -1)
            cv2.putText(frame_rgb, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add FPS counter (development)
        if hasattr(self, '_last_frame_time'):
            fps = 1.0 / (time.time() - self._last_frame_time)
            cv2.putText(frame_rgb, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        self._last_frame_time = time.time()
        
        # Convert to QImage and display
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.camera_label.setPixmap(pixmap)
        
        # Store frame for inference
        self._last_frame = frame
        
    def update_inference(self, result):
        """Update UI with inference results"""
        self.current_product = result['class']
        self.current_confidence = result['confidence']
        
        # Update product display
        product_info = PRODUCT_DB[self.current_product]
        self.product_icon.setText(product_info['icon'])
        self.product_name.setText(product_info['name'])
        self.confidence_bar.setValue(int(self.current_confidence * 100))
        
        # Update price
        self.rate_label.setText(f"₹{product_info['price']:.2f}/kg")
        self.update_total()
        
        # Update status
        if self.current_confidence > 0.9:
            self.status_label.setText(f"✓ {product_info['name']} detected with high confidence")
            self.status_label.setStyleSheet(f"color: {COLOR_SUCCESS}; font-size: 20px; font-weight: bold; padding: 15px; background-color: #E8F5E9; border-radius: 5px; margin: 10px 0;")
        elif self.current_confidence > 0.7:
            self.status_label.setText(f"? {product_info['name']} detected - Please verify")
            self.status_label.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 20px; font-weight: bold; padding: 15px; background-color: #FFF3E0; border-radius: 5px; margin: 10px 0;")
        else:
            self.status_label.setText("Low confidence - Manual selection recommended")
            self.status_label.setStyleSheet(f"color: {COLOR_ERROR}; font-size: 20px; font-weight: bold; padding: 15px; background-color: #FFEBEE; border-radius: 5px; margin: 10px 0;")
            
    def update_weight(self, weight):
        """Update weight display"""
        self.current_weight = weight
        self.weight_label.setText(f"{weight:.3f} kg")
        self.update_total()
        
    def update_total(self):
        """Calculate and update total price"""
        if self.current_product:
            price = PRODUCT_DB[self.current_product]['price']
            total = self.current_weight * price
            self.total_label.setText(f"₹{total:.2f}")
            
    def trigger_inference(self):
        """Trigger inference when weight is stable"""
        if self.current_weight > 0.05 and not self.manual_mode:  # Only run inference if not in manual mode
            if hasattr(self, '_last_frame') and self._last_frame is not None:
                self.inference_thread.process_frame(self._last_frame)
                
    def confirm_transaction(self):
        """Confirm and process the transaction"""
        if self.current_product and self.current_weight > 0:
            product = PRODUCT_DB[self.current_product]
            total = self.current_weight * product['price']
            
            # Log transaction (in real system, this would print receipt)
            print(f"\n{'='*50}")
            print(f"TRANSACTION CONFIRMED")
            print(f"Product: {product['name']}")
            print(f"Weight: {self.current_weight:.3f} kg")
            print(f"Rate: ₹{product['price']:.2f}/kg")
            print(f"Total: ₹{total:.2f}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}\n")
            
            # Log transaction to file
            self.log_transaction(product, total)
            
            # Show confirmation dialog
            QMessageBox.information(self, "Transaction Complete", 
                                  f"Transaction processed!\n\n"
                                  f"{product['name']}: {self.current_weight:.3f} kg\n"
                                  f"Total: ₹{total:.2f}")
            
            self.clear_transaction()
    
    def log_transaction(self, product, total):
        """Log transaction to JSON file"""
        try:
            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Create daily log file
            log_file = log_dir / f"transactions_{datetime.now().strftime('%Y-%m-%d')}.json"
            
            # Transaction data
            transaction = {
                "timestamp": datetime.now().isoformat(),
                "product": product['name'],
                "product_icon": product['icon'],
                "weight_kg": round(self.current_weight, 3),
                "rate_per_kg": product['price'],
                "total": round(total, 2),
                "confidence": round(self.current_confidence, 2)
            }
            
            # Read existing logs or create new list
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Append new transaction
            logs.append(transaction)
            
            # Write back to file
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
            print(f"Transaction logged to: {log_file}")
            
        except Exception as e:
            print(f"Warning: Could not log transaction: {e}")
            # Don't fail the transaction if logging fails
            
    def manual_selection(self):
        """Open manual product selection dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Product")
        dialog.setModal(True)
        layout = QGridLayout()
        
        # Create buttons for each product
        row, col = 0, 0
        for key, product in PRODUCT_DB.items():
            btn = QPushButton(f"{product['icon']}\n{product['name']}\n₹{product['price']}/kg")
            btn.setMinimumSize(180, 180)  # Larger buttons to prevent text truncation
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 20px;
                    padding: 10px;
                }
            """)
            btn.clicked.connect(lambda checked, k=key: self.select_product(k, dialog))
            layout.addWidget(btn, row, col)
            
            col += 1
            if col > 2:
                col = 0
                row += 1
                
        dialog.setLayout(layout)
        dialog.exec_()
        
    def select_product(self, product_key, dialog):
        """Manually select a product"""
        self.manual_mode = True  # Set manual mode to prevent automatic inference
        result = {
            'class': product_key,
            'confidence': 1.0,
            'inference_time': 0,
            'timestamp': datetime.now()
        }
        self.update_inference(result)
        dialog.close()
        
    def clear_transaction(self):
        """Clear current transaction"""
        self.current_product = None
        self.current_confidence = 0.0
        self.manual_mode = False  # Reset manual mode to allow automatic inference
        self.product_icon.setText("🛒")
        self.product_name.setText("Waiting...")
        self.confidence_bar.setValue(0)
        self.rate_label.setText("₹0.00/kg")
        self.total_label.setText("₹0.00")
        self.status_label.setText("Ready - Place item on scale")
        self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 15px; background-color: #EEEEEE; border-radius: 5px; margin: 10px 0;")
        
    def closeEvent(self, event):
        """Clean up threads on close"""
        self.camera_thread.stop()
        self.weight_thread.stop()
        self.inference_thread.terminate()
        
        # Clean up RKNN if it was used
        if hasattr(self, 'use_rknn') and self.use_rknn and hasattr(self, 'rknn_lite'):
            self.rknn_lite.release()
            print("✅ RKNN runtime released")
            
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Clean startup message
    print("\n🎯 AI-Scale POS System Starting...")
    print("================================")
    print("📷 Camera: Initializing...")
    print("🧠 AI Model: Loading...")
    print("⚖️  Scale: Checking serial ports...")
    print("================================\n")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = AIScalePOS()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
