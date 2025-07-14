# AI-Scale PoC: AI-Powered Retail Scale Project

**Project Lead:** Aanand Prabhu – Technical lead of Essae-Teraoka's flagship AI-powered retail scale project.

## Overview

AI-Scale is an intelligent retail weighing system that combines computer vision with traditional weighing scales to automatically identify produce items and calculate prices. The system uses a lightweight MobileNetV3 model optimized for edge deployment on Rockchip RK3568 hardware.

## Current PoC Status (July 2025)

### Model Performance

- **Architecture**: MobileNetV3-Large (ReLU6, ONNX)
- **Classes**: 6 produce types (Apple, Banana, Black Grapes, Mango, Orange, Pomegranate)
- **Dataset**: 1,731 training + 651 validation images
- **Accuracy**: 93.70% Top-1 (F1-scores > 87% for all classes)
- **Model Size**: 16.06MB ONNX → Converted to RKNN format
- **Training Platform**: Google Colab Pro (A100 GPU)

### Development Status

| Component | Status |
|-----------|---------|
| Model Training | ✅ Complete |
| GUI Development | ✅ PyQt5 POS Interface Complete |
| ONNX Integration | ✅ Real model inference working |
| RKNN Conversion | ✅ Complete |
| Hardware Deployment | ⏳ Pending |

## Project Structure

```bash
aiscale/
├── aiscale_pos_gui.py       # Unified POS application (ONNX/RKNN)
├── models/                  # Trained models
│   ├── mobilenetv3_produce.onnx
│   ├── mobilenetv3_produce_simplified.onnx
│   └── mobilenetv3_produce.rknn
├── logs/                    # Transaction logs
├── setup.sh                 # One-click setup & launch
├── requirements.txt         # Python dependencies
└── README.md               # Documentation
```

## Hardware Specifications

### Edge Device: Seavo SV3c-MPOS3568B

- **CPU**: Rockchip RK3568 (Quad A55 @ 2.0GHz)
- **NPU**: 1 TOPS (RKNN Toolkit 2 compatible)
- **RAM**: 4GB LPDDR4X
- **Storage**: 32GB eMMC 5.1
- **OS**: Debian Linux

### Camera: JSK-S8130-V3.0

- **Resolution**: 5MP (2592×1944)
- **Interface**: USB 2.0
- **Capture**: 1920×1080 for performance

### Display: NT156WHM-N42

- **Size**: 15.6" TN Panel
- **Resolution**: 1366×768
- **Interface**: eDP

## GUI Features

The PyQt5-based POS interface includes:

- ✅ Live camera feed with guide brackets
- ✅ Real-time inference (ONNX on PC, RKNN on RK3568)
- ✅ Automatic platform detection
- ✅ Essae scale integration via serial port
- ✅ Weight monitoring with automatic parsing
- ✅ Automatic price calculation (₹/kg)
- ✅ Manual product selection fallback
- ✅ Transaction confirmation and logging
- ✅ High-contrast design for 6-bit displays
- ✅ Quiet mode to reduce terminal output

## Quick Start

### Prerequisites

- Python 3.8+
- OpenCV with camera support
- PyQt5

### One-Click Setup & Launch

```bash
# Clone the repository
git clone <repository-url>
cd Essae_AI_Scale

# Run setup and launch application
./setup.sh
```

The setup script will:

- Create/activate virtual environment
- Install all dependencies
- Launch the AI-Scale POS application

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python aiscale_pos_gui.py
```

### Testing with Webcam

The application supports both mock and real camera modes. To test with a webcam:

1. Ensure camera permissions are granted
2. Point camera at actual produce
3. Observe real-time predictions in the terminal

## Scale Connection

### Connecting Essae Scale

The application supports Essae scales via RS-232 serial communication.

1. **Install serial driver**:

```bash
pip install pyserial
```

2. **Configure serial port** in `aiscale_pos_gui.py` (lines 349-351):

```python
SERIAL_PORT = '/dev/ttyUSB0'  # Linux: /dev/ttyUSB0, Windows: COM3
BAUD_RATE = 9600  # Check your scale manual
```

3. **Common serial ports**

- Linux: `/dev/ttyUSB0`, `/dev/ttyS0`
- Windows: `COM3`, `COM4`
- Mac: `/dev/tty.usbserial-*`

4. **Supported data formats**:

- `ST,GS,+0001.500kg` (status, gross/net, weight)
- `W+0001.500` (weight only)
- `+0001.500kg` (simple format)

The app will:

- Auto-detect available serial ports
- Connect to the scale automatically
- Parse weight data in real-time
- Fall back to mock data if connection fails

## Deployment

### Linux Environment Setup

```bash
# Option A: VirtualBox with Ubuntu 20.04
# Option B: Docker with GUI support
# Option C: UTM for ARM64 testing
```

### RK3568 Hardware Deployment

The application automatically detects and uses RKNN when running on RK3568 hardware.

```bash
# Copy to RK3568 device
scp -r aiscale_pos_gui.py models/ requirements.txt user@rk3568:/opt/aiscale/

# On the RK3568 device:
cd /opt/aiscale
pip install numpy opencv-python PyQt5
# Note: rknnlite should be pre-installed on RK3568

# Run the same POS application
python aiscale_pos_gui.py
```

The application will automatically:

- Detect RKNNLite availability
- Load the RKNN model instead of ONNX
- Display "🔧 RKNNLite detected - using RKNN inference" in console

## Next Steps

| Task | Status | Priority | Time Estimate |
|------|--------|----------|---------------|
| Test with real produce (webcam) | Ready | High | 30 mins |
| Deploy to RK3568 hardware | Ready | High | 1 hour |
| Connect JSK camera | Ready | Medium | 30 mins |
| Serial weight integration | ✅ Complete | Medium | Done |
| Configure scale serial port | Ready | Medium | 10 mins |
| End-to-end hardware testing | Ready | High | Half-day |

## Development Path

1. ✅ Model Training (Complete)
2. ✅ GUI Development (Complete)
3. ✅ ONNX Integration (Complete)
4. ✅ RKNN Conversion (Complete)
5. ⏳ Hardware Deployment (Ready)
6. ⏳ Production Integration
