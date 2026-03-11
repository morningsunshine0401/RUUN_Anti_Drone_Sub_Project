# Real-time Small Drone Tracking System

High-performance drone tracking on NVIDIA Jetson using YOLO + DaSiamRPN with sparse tracking optimization.

![System Performance](https://img.shields.io/badge/FPS-30+-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano-orange)

## 🎥 Demo

https://github.com/user-attachments/assets/66a58aed-6267-4cb4-b6aa-cbfe8318cc64


> Real-time tracking of small drones (5×5 pixels) at 30+ FPS

## 🎯 Key Features

- **Small Object Tracking**: Handles drones as small as 5×5 pixels
- **Real-time Performance**: 30~45+ FPS on Jetson Orin Nano
- **Sparse Tracking**: 2.7× speed improvement over dense tracking
- **Hybrid Approach**: YOLO detection + DaSiamRPN tracking
- **Manual Initialization**: Human-in-the-loop for distant targets

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/morningsunshine0401/RUUN_Anti_Drone_Sub_Project.git
cd RUUN_Anti_Drone_Sub_Project

# Install dependencies
pip install torch torchvision ultralytics opencv-python numpy --break-system-packages

# Setup DaSiamRPN
git clone https://github.com/foolwood/DaSiamRPN.git
cd DaSiamRPN/code
wget http://www.robots.ox.ac.uk/~qwang/SiamRPNVOT.model
cd ../..
```

### Run

```bash
# Basic usage
python3 jetson_sparse_dasiamrpn_v3_FIN.py \
    --model best_fp16.engine \
    --source videos/test11_640.mp4

# With manual initialization (for small/distant drones)
python3 jetson_sparse_dasiamrpn_v3_FIN.py \
    --model best_fp16.engine \
    --source videos/test11_640.mp4 \
    --manual-init \
    --tracking-interval 3 \
    --jump-threshold 400
```

## 🏗️ Architecture

```
Input Frame
    ↓
┌───────────────┬───────────────┐
↓               ↓               ↓
YOLO       DaSiamRPN        Cache
(Every)    (Sparse)        (Reuse)
13-18ms     0-35ms          0ms
└───────────────┴───────────────┘
    ↓
Jump Detection (False Positive Filter)
    ↓
Final Result (30+ FPS)
```

**Concept**: 
- YOLO provides accurate detection every frame
- DaSiamRPN tracks when YOLO fails or validates detections
- Sparse execution: Run DaSiamRPN every N frames, cache results between
- Jump detection rejects sudden false positives

## 📊 Performance

- **FPS**: 30-45 on Jetson Orin Nano
- **Detection Coverage**: 85%
- **Tracking Coverage**: 13% (fills gaps)
- **False Positives**: <2%

## 🎛️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | required | YOLO model path |
| `--source` | `0` | Video file or camera ID |
| `--tracking-interval` | `3` | Run DaSiamRPN every N frames |
| `--jump-threshold` | `200` | Max detection jump (pixels) |
| `--conf` | `0.5` | YOLO confidence threshold |
| `--manual-init` | `False` | Enable manual bbox selection |

### Tuning Tips

**For speed** (60+ FPS):
```bash
--tracking-interval 5 --jump-threshold 300
```

**For accuracy** (40-45 FPS):
```bash
--tracking-interval 2 --jump-threshold 150
```

**For small/distant targets**:
```bash
--manual-init --conf 0.3
```

## 🔬 Technical Approach

### Problem
Tracking small drones (5×5 pixels) in real-time on edge devices.

### Solution
1. **Hybrid Detection-Tracking**: Combine YOLO (drift-free) with DaSiamRPN (gap-filling)
2. **Sparse Tracking**: Run expensive tracker selectively, cache results
3. **Jump Detection**: Validate detections using tracking for consistency
4. **Bbox Expansion**: Ensure minimum 24×24 pixels for tracker initialization
5. **Manual Init**: Human assistance for extremely small targets

### Why DaSiamRPN?
- Deep learning tracker with scale-invariant features
- Works on objects as small as 5×5 pixels (vs KCF: 20%, CSRT: 40%)
- Template matching approach prevents drift
- BUT slow (35ms) → solved with sparse execution

## 📁 Project Structure

```
drone-tracking/
├── jetson_sparse_dasiamrpn_v3_FIN.py
├── videos/
│   └── test11_640.mp4
├── DaSiamRPN/
│   └── code/
│       └── SiamRPNVOT.model
└── README.md
```

## 🐛 Troubleshooting

**Low FPS**: Increase `--tracking-interval` or use TensorRT model

**Tracking failures**: Lower `--conf` threshold or use `--manual-init`

**False positives**: Increase `--conf` or decrease `--jump-threshold`

## 📚 References

- [DaSiamRPN](https://arxiv.org/abs/1808.06048)
- [YOLO](https://github.com/ultralytics/ultralytics)

## 📧 Contact

- Email: runbk0401@naver.com
- Issues: [GitHub Issues](https://github.com/morningsunshine0401/RUUN_Anti_Drone_Sub_Project/issues)

---

**MIT License** • Made for real-time edge AI
