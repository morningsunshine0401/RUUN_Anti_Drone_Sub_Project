#!/usr/bin/env python3
"""
YOLO + Tracker 통합 테스트
- Detector 우선 (YOLO)
- Detector 실패 시 Tracker 사용
- Tracker: opencv(CSRT) 또는 DaSiamRPN

필요:
- ultralytics
- opencv-python
- torch
- DaSiamRPN 코드(현재 폴더/패키지로 import 가능해야 함):
    net.py (SiamRPNvot)
    run_SiamRPN.py (SiamRPN_init, SiamRPN_track)
    utils.py (optional: get_axis_aligned_bbox 등)
    SiamRPNVOT.model (가중치 파일)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
import time


# -----------------------------
# DaSiamRPN imports (필수)
# -----------------------------
try:
    import torch
    from os.path import realpath, dirname, join
    from net import SiamRPNvot
    from run_SiamRPN import SiamRPN_init, SiamRPN_track
except Exception as e:
    SiamRPNvot = None
    SiamRPN_init = None
    SiamRPN_track = None
    torch = None
    DASIAM_IMPORT_ERROR = e
else:
    DASIAM_IMPORT_ERROR = None


class DroneDetectorTracker:
    """
    D-bias 방식:
    - 탐지(Detector) 우선
    - 탐지 실패 시 추적(Tracker)
    """

    def __init__(self, detector_path, tracker_type='opencv', conf_threshold=0.5,
                 dasiam_model_path=None, dasiam_use_cuda=True):
        """
        Args:
            detector_path: YOLO 모델 경로
            tracker_type: 'opencv' | 'dasiamrpn'
            conf_threshold: YOLO conf threshold
            dasiam_model_path: SiamRPNVOT.model 경로 (tracker_type='dasiamrpn'일 때 필요)
            dasiam_use_cuda: CUDA 사용 여부
        """
        print(f"🔍 YOLO 로드: {detector_path}")
        self.detector = YOLO(detector_path)
        self.conf_threshold = conf_threshold

        self.tracker_type = tracker_type
        self.tracker = None
        self.tracking_initialized = False

        # DaSiamRPN state/net
        self.dasiam_net = None
        self.dasiam_state = None
        self.dasiam_use_cuda = dasiam_use_cuda

        if self.tracker_type == 'dasiamrpn':
            self._init_dasiamrpn_net(dasiam_model_path)

        print(f"✅ 초기화 완료 (추적기: {tracker_type})")

    # -----------------------------
    # Tracker init
    # -----------------------------
    def init_tracker(self, frame, bbox_xywh):
        """
        bbox_xywh: [x, y, w, h] (int 권장)
        """
        if bbox_xywh is None:
            self.tracking_initialized = False
            return

        x, y, w, h = bbox_xywh
        # 너무 작은 bbox면 tracker init 실패 확률 높음
        if w < 8 or h < 8:
            print(f"⚠️ bbox too small for init: {bbox_xywh} -> tracking off")
            self.tracking_initialized = False
            return

        if self.tracker:
            pass

        if self.tracker_type == 'opencv':
            self.tracker = cv2.TrackerCSRT_create()
            # OpenCV는 tuple(float)도 되긴 하지만, 안전하게 float로 변환
            self.tracker.init(frame, (float(x), float(y), float(w), float(h)))
            self.tracking_initialized = True
            print("📍 OpenCV CSRT 추적기 초기화")

        elif self.tracker_type == 'dasiamrpn':
            if self.dasiam_net is None:
                print("❌ DaSiamRPN net not loaded.")
                self.tracking_initialized = False
                return

            # DaSiamRPN init은 (target_pos=cxcy, target_sz=wh)
            cx = x + w / 2.0
            cy = y + h / 2.0
            target_pos = np.array([cx, cy], dtype=np.float32)
            target_sz = np.array([w, h], dtype=np.float32)

            # state 생성
            self.dasiam_state = SiamRPN_init(frame, target_pos, target_sz, self.dasiam_net)
            self.tracking_initialized = True
            print("📍 DaSiamRPN 추적기 초기화")

        else:
            raise ValueError(f"Unknown tracker_type: {self.tracker_type}")

    # -----------------------------
    # YOLO detect
    # -----------------------------
    def detect(self, frame):
        """
        Returns:
            bbox_xywh: [x, y, w, h] or None
            conf: float
        """
        results = self.detector.predict(frame, conf=self.conf_threshold, verbose=False)

        if len(results) == 0:
            return None, 0.0

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, 0.0

        # 가장 높은 conf의 박스를 고르기 (0번이 항상 최고는 아닐 수 있음)
        confs = boxes.conf.detach().cpu().numpy()
        best_i = int(np.argmax(confs))
        box = boxes[best_i]

        xyxy = box.xyxy[0].detach().cpu().numpy()
        conf = float(box.conf[0].detach().cpu().numpy())

        x1, y1, x2, y2 = xyxy
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

        # bbox sanity
        if w <= 0 or h <= 0:
            return None, 0.0

        return [x, y, w, h], conf

    # -----------------------------
    # Tracker update
    # -----------------------------
    def track(self, frame):
        """
        Returns:
            bbox_xywh or None
        """
        if not self.tracking_initialized:
            return None

        if self.tracker_type == 'opencv':
            if self.tracker is None:
                return None
            success, bbox = self.tracker.update(frame)
            if success:
                x, y, w, h = bbox
                return [int(x), int(y), int(w), int(h)]
            else:
                print("⚠️ OpenCV tracker 실패")
                self.tracking_initialized = False
                return None

        elif self.tracker_type == 'dasiamrpn':
            if self.dasiam_state is None:
                return None

            # SiamRPN_track returns updated state with 'target_pos','target_sz'
            self.dasiam_state = SiamRPN_track(self.dasiam_state, frame)

            pos = self.dasiam_state.get('target_pos', None)
            sz = self.dasiam_state.get('target_sz', None)
            if pos is None or sz is None:
                print("⚠️ DaSiamRPN state invalid")
                self.tracking_initialized = False
                return None

            cx, cy = float(pos[0]), float(pos[1])
            w, h = float(sz[0]), float(sz[1])

            # 너무 작은 타겟은 실패로 처리
            if w < 4 or h < 4 or not np.isfinite([cx, cy, w, h]).all():
                print("⚠️ DaSiamRPN invalid bbox")
                self.tracking_initialized = False
                return None

            x = int(cx - w / 2.0)
            y = int(cy - h / 2.0)
            return [x, y, int(w), int(h)]

        else:
            return None

    # -----------------------------
    # D-bias process
    # -----------------------------
    def process_frame(self, frame):
        """
        Returns:
            bbox, status, conf
        """
        # 1) 탐지 시도
        bbox, conf = self.detect(frame)
        if bbox is not None:
            # 탐지 성공: tracker re-init
            self.init_tracker(frame, bbox)
            return bbox, 'DETECTED', conf

        # 2) 탐지 실패: tracker로 추적
        bbox = self.track(frame)
        if bbox is not None:
            return bbox, 'TRACKED', 0.0

        # 3) 모두 실패
        return None, 'LOST', 0.0

    # -----------------------------
    # DaSiamRPN net load
    # -----------------------------
    def _init_dasiamrpn_net(self, model_path):
        if SiamRPNvot is None or SiamRPN_init is None or SiamRPN_track is None:
            print("❌ DaSiamRPN import 실패:", DASIAM_IMPORT_ERROR)
            raise RuntimeError("DaSiamRPN 모듈 import가 안됩니다. (net/run_SiamRPN/utils 확인)")

        if model_path is None:
            # 기본: 현재 파일 기준 같은 폴더에 SiamRPNVOT.model 있다고 가정
            model_path = join(realpath(dirname(__file__)), 'SiamRPNVOT.model')

        model_path = str(model_path)
        print(f"🧠 DaSiamRPN 로드: {model_path}")

        self.dasiam_net = SiamRPNvot()

        # weights load
        state = torch.load(model_path, map_location='cpu')
        self.dasiam_net.load_state_dict(state)
        self.dasiam_net.eval()

        if self.dasiam_use_cuda and torch.cuda.is_available():
            self.dasiam_net = self.dasiam_net.cuda()
            print("🚀 DaSiamRPN: CUDA 사용")
        else:
            print("🧊 DaSiamRPN: CPU 사용 (CUDA 비활성/불가)")

        # tracker state는 init에서 생성
        self.dasiam_state = None
        self.tracking_initialized = False


def draw_bbox(frame, bbox, status, conf=0.0):
    if bbox is None:
        return frame

    x, y, w, h = bbox

    colors = {
        'DETECTED': (0, 255, 0),
        'TRACKED': (255, 165, 0),
        'LOST': (0, 0, 255)
    }
    color = colors.get(status, (255, 255, 255))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"{status}"
    if conf > 0:
        label += f" {conf:.2f}"
    cv2.putText(frame, label, (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def test_on_video(detector_tracker, video_path, output_path=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 비디오 열기 실패: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 비디오: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        bbox, status, conf = detector_tracker.process_frame(frame)

        frame = draw_bbox(frame, bbox, status, conf)

        elapsed = time.time() - t0
        curr_fps = frame_idx / max(elapsed, 1e-6)
        status_text = f"Frame {frame_idx}/{total_frames} | {status} | FPS {curr_fps:.1f}"
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Drone Detection & Tracking', frame)
        if out is not None:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"✅ 완료: {frame_idx} frames")


def test_on_webcam(detector_tracker):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    print("📹 웹캠 테스트 시작 (ESC로 종료)")

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        bbox, status, conf = detector_tracker.process_frame(frame)

        frame = draw_bbox(frame, bbox, status, conf)

        elapsed = time.time() - t0
        curr_fps = frame_idx / max(elapsed, 1e-6)
        cv2.putText(frame, f"{status} | FPS {curr_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Drone Detection & Tracking (Webcam)', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='YOLO + Tracker 통합 테스트')

    parser.add_argument('--model', type=str, required=True,
                        help='YOLO 모델 경로 (best.pt)')
    parser.add_argument('--video', type=str, help='테스트 비디오 경로')
    parser.add_argument('--webcam', action='store_true', help='웹캠 사용')
    parser.add_argument('--output', type=str, help='결과 비디오 저장 경로')
    parser.add_argument('--conf', type=float, default=0.5, help='탐지 신뢰도 임계값')

    parser.add_argument('--tracker', type=str, default='opencv',
                        choices=['opencv', 'dasiamrpn'],
                        help="추적기 종류: opencv | dasiamrpn")

    parser.add_argument('--dasiam_model', type=str, default=None,
                        help="DaSiamRPN weight 경로 (SiamRPNVOT.model)")
    parser.add_argument('--dasiam_cpu', action='store_true',
                        help="DaSiamRPN을 CPU로 강제 실행")

    args = parser.parse_args()

    detector_tracker = DroneDetectorTracker(
        detector_path=args.model,
        tracker_type=args.tracker,
        conf_threshold=args.conf,
        dasiam_model_path=args.dasiam_model,
        dasiam_use_cuda=(not args.dasiam_cpu)
    )

    if args.webcam:
        test_on_webcam(detector_tracker)
    elif args.video:
        test_on_video(detector_tracker, args.video, args.output)
    else:
        print("❌ --video 또는 --webcam 중 하나를 선택하세요")


if __name__ == "__main__":
    main()
