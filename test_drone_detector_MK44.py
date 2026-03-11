#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO + Tracker 통합 테스트
- Detector 우선 (YOLO)
- Detector 실패 시 Tracker 사용
- Tracker: opencv(CSRT) 또는 DaSiamRPN

추가 기능:
- --manual_init: 비디오 시작 시 수동 bbox 선택
  - 'n' 키로 한 프레임씩 넘기기
  - 's' 키로 ROI 선택 시작
  - 'q' 키로 수동 init 포기하고 YOLO 자동 시작
- OpenCV 4.12 bbox 타입 문제 해결 (legacy tracker + python int 강제 변환)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time

# -----------------------------
# DaSiamRPN imports (선택)
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


def _as_pyint_bbox(b):
    """bbox를 무조건 파이썬 int 4개로 강제 변환"""
    if b is None:
        return None
    if isinstance(b, (tuple, list)) and len(b) == 4:
        x, y, w, h = b
        # numpy scalar -> python scalar
        x = int(np.asarray(x).item())
        y = int(np.asarray(y).item())
        w = int(np.asarray(w).item())
        h = int(np.asarray(h).item())
        return [x, y, w, h]
    raise ValueError("bbox must be list/tuple of 4 numbers")


def _make_csrt_tracker():
    """
    OpenCV 4.12에서 tracker API가 legacy에 있는 경우가 많아서
    legacy 우선 사용.
    """
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker is not available. Install opencv-contrib-python.")


class DroneDetectorTracker:
    def __init__(self,
                 detector_path,
                 tracker_type='opencv',
                 conf_threshold=0.5,
                 dasiam_model_path=None,
                 dasiam_use_cuda=True,
                 # re-init 안정화(작은 bbox여도 추적 가능하게)
                 reinit_iou=0.2,
                 reinit_interval=10,
                 min_wh=32):
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

        # 안정화 파라미터
        self.reinit_iou = float(reinit_iou)
        self.reinit_interval = int(reinit_interval)
        self.min_wh = int(min_wh)
        self.frame_idx = 0
        self.last_init_frame = -10**9

        if self.tracker_type == 'dasiamrpn':
            self._init_dasiamrpn_net(dasiam_model_path)

        print(f"✅ 초기화 완료 (추적기: {tracker_type})")

    def _iou_xywh(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _expand_bbox_keep_center(self, bbox, W, H):
        """
        bbox가 작아도 추적 가능하게, init용 bbox는 중심 유지 + 최소 크기(min_wh)로 확장
        (작은 물체를 무시하는게 아니라, tracker가 먹을 컨텍스트 확보)
        """
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        new_w = max(w, self.min_wh)
        new_h = max(h, self.min_wh)
        nx = int(round(cx - new_w / 2.0))
        ny = int(round(cy - new_h / 2.0))
        nx = max(0, min(nx, W - 1))
        ny = max(0, min(ny, H - 1))
        nx2 = min(W - 1, nx + int(new_w))
        ny2 = min(H - 1, ny + int(new_h))
        return [nx, ny, max(1, nx2 - nx), max(1, ny2 - ny)]

    def init_tracker(self, frame, bbox_xywh):
        bbox_xywh = _as_pyint_bbox(bbox_xywh)
        x, y, w, h = bbox_xywh

        if w <= 1 or h <= 1:
            self.tracking_initialized = False
            return

        if self.tracker_type == 'opencv':
            self.tracker = _make_csrt_tracker()
            # ✅ OpenCV 4.12 parsing 에러 방지: (int,int,int,int)로 넣기
            self.tracker.init(frame, (int(x), int(y), int(w), int(h)))
            self.tracking_initialized = True
            # print("📍 OpenCV CSRT 추적기 초기화")

        elif self.tracker_type == 'dasiamrpn':
            if self.dasiam_net is None:
                print("❌ DaSiamRPN net not loaded.")
                self.tracking_initialized = False
                return
            cx = x + w / 2.0
            cy = y + h / 2.0
            target_pos = np.array([cx, cy], dtype=np.float32)
            target_sz = np.array([w, h], dtype=np.float32)
            self.dasiam_state = SiamRPN_init(frame, target_pos, target_sz, self.dasiam_net)
            self.tracking_initialized = True
            # print("📍 DaSiamRPN 추적기 초기화")
        else:
            raise ValueError(f"Unknown tracker_type: {self.tracker_type}")

    def detect(self, frame):
        results = self.detector.predict(frame, conf=self.conf_threshold, verbose=False)
        if len(results) == 0:
            return None, 0.0
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, 0.0

        confs = boxes.conf.detach().cpu().numpy()
        best_i = int(np.argmax(confs))
        box = boxes[best_i]
        xyxy = box.xyxy[0].detach().cpu().numpy()
        conf = float(box.conf[0].detach().cpu().numpy())

        x1, y1, x2, y2 = xyxy
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        if w <= 0 or h <= 0:
            return None, 0.0
        return [x, y, w, h], conf

    def track(self, frame):
        if not self.tracking_initialized:
            return None

        if self.tracker_type == 'opencv':
            if self.tracker is None:
                return None
            ok, bbox = self.tracker.update(frame)
            if not ok:
                self.tracking_initialized = False
                return None
            x, y, w, h = bbox
            return _as_pyint_bbox([x, y, w, h])

        elif self.tracker_type == 'dasiamrpn':
            if self.dasiam_state is None:
                return None
            self.dasiam_state = SiamRPN_track(self.dasiam_state, frame)
            pos = self.dasiam_state.get('target_pos', None)
            sz = self.dasiam_state.get('target_sz', None)
            if pos is None or sz is None:
                self.tracking_initialized = False
                return None
            cx, cy = float(pos[0]), float(pos[1])
            w, h = float(sz[0]), float(sz[1])
            if not np.isfinite([cx, cy, w, h]).all():
                self.tracking_initialized = False
                return None
            x = int(cx - w / 2.0)
            y = int(cy - h / 2.0)
            return _as_pyint_bbox([x, y, int(w), int(h)])

        return None

    def process_frame(self, frame):
        self.frame_idx += 1
        H, W = frame.shape[:2]

        det_bbox, conf = self.detect(frame)

        # 먼저 tracking 업데이트(가능하면)
        trk_bbox = None
        if self.tracking_initialized:
            trk_bbox = self.track(frame)

        # 탐지가 있으면: "조건부 re-init" (매 프레임 init 금지)
        if det_bbox is not None:
            # 작은 bbox도 추적 가능하게 init bbox는 컨텍스트 확보용으로 확장
            init_bbox = self._expand_bbox_keep_center(det_bbox, W, H)

            should_reinit = False
            if (not self.tracking_initialized) or (trk_bbox is None):
                should_reinit = True
            else:
                if (self.frame_idx - self.last_init_frame) >= self.reinit_interval:
                    iou = self._iou_xywh(init_bbox, trk_bbox)
                    if iou < self.reinit_iou:
                        should_reinit = True

            if should_reinit:
                self.init_tracker(frame, init_bbox)
                self.last_init_frame = self.frame_idx
                return det_bbox, 'DETECTED', conf

            if trk_bbox is not None:
                return trk_bbox, 'TRACKED', 0.0
            return det_bbox, 'DETECTED', conf

        # 탐지 없으면 tracking만
        if trk_bbox is not None:
            return trk_bbox, 'TRACKED', 0.0

        return None, 'LOST', 0.0

    def _init_dasiamrpn_net(self, model_path):
        if SiamRPNvot is None or SiamRPN_init is None or SiamRPN_track is None:
            print("❌ DaSiamRPN import 실패:", DASIAM_IMPORT_ERROR)
            raise RuntimeError("DaSiamRPN 모듈 import가 안됩니다. (net/run_SiamRPN/utils 확인)")

        if model_path is None:
            model_path = join(realpath(dirname(__file__)), 'SiamRPNVOT.model')

        model_path = str(model_path)
        print(f"🧠 DaSiamRPN 로드: {model_path}")

        self.dasiam_net = SiamRPNvot()
        state = torch.load(model_path, map_location='cpu')
        self.dasiam_net.load_state_dict(state)
        self.dasiam_net.eval()

        if self.dasiam_use_cuda and torch.cuda.is_available():
            self.dasiam_net = self.dasiam_net.cuda()
            print("🚀 DaSiamRPN: CUDA 사용")
        else:
            print("🧊 DaSiamRPN: CPU 사용")

        self.dasiam_state = None
        self.tracking_initialized = False


def draw_bbox(frame, bbox, status, conf=0.0):
    if bbox is None:
        return frame
    x, y, w, h = bbox
    colors = {'DETECTED': (0, 255, 0), 'TRACKED': (255, 165, 0), 'LOST': (0, 0, 255)}
    color = colors.get(status, (255, 255, 255))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = status if conf <= 0 else f"{status} {conf:.2f}"
    cv2.putText(frame, label, (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def manual_init_with_skip(cap):
    """
    첫 부분에서 원하는 프레임까지 n키로 넘기고,
    s키로 ROI 선택을 시작.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            return None, None  # 실패

        disp = frame.copy()
        cv2.putText(disp, "Manual init: [n]=next frame  [s]=select ROI  [q]=skip manual",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Manual Init", disp)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            continue
        if key == ord('q'):
            cv2.destroyWindow("Manual Init")
            return frame, None  # manual 포기
        if key == ord('s'):
            roi = cv2.selectROI("Select ROI (ENTER/SPACE confirm, c cancel)",
                                frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select ROI (ENTER/SPACE confirm, c cancel)")
            cv2.destroyWindow("Manual Init")
            x, y, w, h = roi
            if w == 0 or h == 0:
                return frame, None
            return frame, [int(x), int(y), int(w), int(h)]
        if key == 27:  # ESC
            cv2.destroyWindow("Manual Init")
            return frame, None


def test_on_video(detector_tracker, video_path, output_path=None, manual_init=False):
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

    # 수동 init
    if manual_init:
        first_frame, roi_bbox = manual_init_with_skip(cap)
        if first_frame is None:
            print("❌ 수동 init 프레임 획득 실패")
            cap.release()
            return
        if roi_bbox is not None:
            print(f"✅ 수동 초기 bbox: {roi_bbox}")
            detector_tracker.init_tracker(first_frame, roi_bbox)
        else:
            print("⚠️ 수동 bbox 없음 -> 자동(YOLO) 시작")

        # 다시 0프레임으로 돌리지 말고, 지금 위치에서 계속 진행하는 게 자연스러움
        # (원하면 아래 주석 해제하면 처음부터 다시 읽음)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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

        cv2.imshow("Drone Detection & Tracking", frame)
        if out is not None:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"✅ 완료: {frame_idx} frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--video', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--conf', type=float, default=0.5)

    parser.add_argument('--tracker', type=str, default='opencv',
                        choices=['opencv', 'dasiamrpn'])

    parser.add_argument('--dasiam_model', type=str, default=None)
    parser.add_argument('--dasiam_cpu', action='store_true')

    parser.add_argument('--manual_init', action='store_true')

    # 안정화 파라미터
    parser.add_argument('--reinit_iou', type=float, default=0.2)
    parser.add_argument('--reinit_interval', type=int, default=10)
    parser.add_argument('--min_wh', type=int, default=32)

    args = parser.parse_args()

    detector_tracker = DroneDetectorTracker(
        detector_path=args.model,
        tracker_type=args.tracker,
        conf_threshold=args.conf,
        dasiam_model_path=args.dasiam_model,
        dasiam_use_cuda=(not args.dasiam_cpu),
        reinit_iou=args.reinit_iou,
        reinit_interval=args.reinit_interval,
        min_wh=args.min_wh
    )

    if args.video:
        test_on_video(detector_tracker, args.video, args.output, manual_init=args.manual_init)
    else:
        print("❌ --video를 지정하세요")


if __name__ == "__main__":
    main()
