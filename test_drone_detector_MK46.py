#!/usr/bin/env python3
"""
MK46: Tracking-first (CSRT/DaSiamRPN) + Periodic YOLO correction + (optional) Kalman gating

Why this version:
- Jetson에서 매 프레임 YOLO 돌리면 느리고, 가끔 false positive가 트랙을 갈아타게 만들 수 있음
- 기본은 tracker로 연속 추적 (fast)
- YOLO는 (1) N프레임마다, (2) 트랙 품질이 나쁠 때만 실행해서 drift 교정 (slow/corrector)
- YOLO 결과는 "게이팅(gating)" 통과할 때만 채택 (false positive 방지)
- (optional) Kalman filter로 예측 중심을 만들고 게이팅을 더 안정적으로 수행

Keys:
- 시작 수동 init: --manual_init (n: 프레임 스킵, s: ROI 선택, q: 취소)
- 재생 중: r (ROI 다시 선택 + YOLO snap + tracker 재초기화), p (pause), ESC (exit)

Trackers:
- opencv: CSRT
- dasiamrpn: DaSiamRPN (SiamRPNVOT.model 필요, DaSiamRPN 코드 폴더 필요)

Example:
python test_drone_detector_MK46.py --model best.pt --video test.mp4 --tracker opencv --manual_init
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import math


# -----------------------
# Utils
# -----------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def xywh_to_xyxy(b):
    x, y, w, h = b
    return (x, y, x + w, y + h)

def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return (x1, y1, x2 - x1, y2 - y1)

def iou_xywh(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter + 1e-9
    return float(inter / denom)

def center_of_xywh(b):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)

def expand_min_wh(b, min_wh):
    """Ensure bbox w,h >= min_wh by expanding about center."""
    x, y, w, h = b
    cx, cy = x + w / 2.0, y + h / 2.0
    w2 = max(w, min_wh)
    h2 = max(h, min_wh)
    x2 = cx - w2 / 2.0
    y2 = cy - h2 / 2.0
    return [int(round(x2)), int(round(y2)), int(round(w2)), int(round(h2))]

def clip_bbox_to_frame(b, W, H):
    x, y, w, h = b
    x = int(clamp(x, 0, W - 1))
    y = int(clamp(y, 0, H - 1))
    w = int(clamp(w, 1, W - x))
    h = int(clamp(h, 1, H - y))
    return [x, y, w, h]

def crop_roi(frame, roi_xywh):
    x, y, w, h = roi_xywh
    return frame[y:y+h, x:x+w]

def map_bbox_from_roi(b_roi, roi_xywh):
    x, y, w, h = roi_xywh
    bx, by, bw, bh = b_roi
    return [bx + x, by + y, bw, bh]

def best_overlap(det_bboxes, ref_bbox):
    """Pick detection bbox with best IoU to ref_bbox."""
    if not det_bboxes:
        return None, 0.0
    best_b = None
    best_i = -1.0
    for b in det_bboxes:
        v = iou_xywh(b, ref_bbox)
        if v > best_i:
            best_i = v
            best_b = b
    return best_b, float(best_i)

# -----------------------
# Simple Kalman Filter (constant velocity in image space for center)
# State: [cx, cy, vx, vy]^T
# Measure: [cx, cy]^T
# -----------------------
class SimpleKF:
    def __init__(self):
        self.x = None  # (4,1)
        self.P = None  # (4,4)
        self.Q = None  # process noise
        self.R = None  # measurement noise
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.last_dt = 1.0

    def reset(self, cx, cy, vx=0.0, vy=0.0, pos_var=50.0, vel_var=200.0):
        self.x = np.array([[cx],[cy],[vx],[vy]], dtype=np.float32)
        self.P = np.diag([pos_var, pos_var, vel_var, vel_var]).astype(np.float32)
        # default noises (can be tuned)
        self.Q = np.diag([3.0, 3.0, 20.0, 20.0]).astype(np.float32)
        self.R = np.diag([25.0, 25.0]).astype(np.float32)

    def predict(self, dt=1.0):
        if self.x is None:
            return None
        self.last_dt = float(dt)
        F = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1,0],
                      [0,0,0,1]], dtype=np.float32)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, meas_cx, meas_cy):
        if self.x is None:
            self.reset(meas_cx, meas_cy)
            return self.x.copy()
        z = np.array([[meas_cx],[meas_cy]], dtype=np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()

    def get_center(self):
        if self.x is None:
            return None
        return float(self.x[0,0]), float(self.x[1,0])

# -----------------------
# DaSiamRPN wrapper (optional)
# -----------------------
class DaSiamRPNWrapper:
    def __init__(self, model_path: str):
        # Lazy import to avoid breaking opencv-only usage
        import torch
        from os.path import realpath, dirname, join
        from net import SiamRPNvot
        from run_SiamRPN import SiamRPN_init, SiamRPN_track

        self.torch = torch
        self.SiamRPN_init = SiamRPN_init
        self.SiamRPN_track = SiamRPN_track

        net = SiamRPNvot()
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        net.eval().cuda()
        self.net = net

        self.state = None

    def init(self, frame_bgr, bbox_xywh):
        # DaSiam expects target_pos, target_sz
        x, y, w, h = bbox_xywh
        cx = x + w / 2.0
        cy = y + h / 2.0
        target_pos = np.array([cx, cy], dtype=np.float32)
        target_sz = np.array([w, h], dtype=np.float32)
        self.state = self.SiamRPN_init(frame_bgr, target_pos, target_sz, self.net)
        return True

    def update(self, frame_bgr):
        if self.state is None:
            return False, None
        self.state = self.SiamRPN_track(self.state, frame_bgr)
        # state['target_pos'], state['target_sz'] -> xywh
        tp = self.state["target_pos"]
        ts = self.state["target_sz"]
        cx, cy = float(tp[0]), float(tp[1])
        w, h = float(ts[0]), float(ts[1])
        x = int(round(cx - w / 2.0))
        y = int(round(cy - h / 2.0))
        return True, [x, y, int(round(w)), int(round(h))]

# -----------------------
# Main Detector-Tracker
# -----------------------
@dataclass
class TrackConfig:
    conf_threshold: float = 0.5
    det_period: int = 15               # run YOLO every N frames when tracking ok
    det_on_lost: bool = True           # run YOLO when tracking lost/low-quality
    roi_scale: float = 3.0             # YOLO runs on ROI around predicted bbox
    snap_iou_min: float = 0.05         # manual ROI -> YOLO snap minimum IoU
    gate_center_px: float = 140.0      # gating: max center distance from predicted
    gate_iou_min: float = 0.02         # gating: min IoU with predicted bbox
    min_wh: int = 16                   # allow small, but not zero
    reinit_interval: int = 10          # check reinit every N frames when YOLO correction happens
    reinit_iou: float = 0.30           # if det differs too much from tracker, reinit
    use_kf: bool = True               # Kalman gating predictor
    kf_dt: float = 1.0                # dt per frame in KF (fine)
    debug: bool = False

class DroneDetectorTracker:
    def __init__(self, detector_path: str, tracker_type: str, cfg: TrackConfig,
                 dasiam_model_path: str = None):
        print(f"🔍 YOLO 로드: {detector_path}")
        self.detector = YOLO(detector_path)
        self.tracker_type = tracker_type
        self.cfg = cfg

        self.tracking_initialized = False
        self.tracker = None
        self.dasiam = None
        if tracker_type == "dasiamrpn":
            if not dasiam_model_path:
                raise ValueError("--dasiam_model is required for tracker= d asiamrpn")
            print(f"🧠 DaSiamRPN 로드: {dasiam_model_path}")
            self.dasiam = DaSiamRPNWrapper(dasiam_model_path)

        self.kf = SimpleKF() if cfg.use_kf else None

        self.frame_idx = 0
        self.last_bbox = None          # last output bbox
        self.last_track_bbox = None    # last tracker bbox
        self.last_det_bbox = None      # last detector bbox
        self.last_conf = 0.0

        print(f"✅ 초기화 완료 (tracker={tracker_type}, det_period={cfg.det_period}, use_kf={cfg.use_kf})")

    # -------- Tracker init/update --------
    def init_tracker(self, frame, bbox_xywh):
        H, W = frame.shape[:2]
        b = expand_min_wh(bbox_xywh, self.cfg.min_wh)
        b = clip_bbox_to_frame(b, W, H)

        if self.tracker_type == "opencv":
            self.tracker = cv2.TrackerCSRT_create()
            # OpenCV expects tuple of python floats or ints (avoid numpy scalar)
            self.tracker.init(frame, (int(b[0]), int(b[1]), int(b[2]), int(b[3])))
            self.tracking_initialized = True
            print("📍 CSRT 추적기 초기화:", b)

        elif self.tracker_type == "dasiamrpn":
            ok = self.dasiam.init(frame, b)
            self.tracking_initialized = bool(ok)
            print("📍 DaSiamRPN 추적기 초기화:", b)

        else:
            raise ValueError(f"Unknown tracker_type: {self.tracker_type}")

        # KF reset
        if self.kf is not None:
            cx, cy = center_of_xywh(b)
            self.kf.reset(cx, cy)

        self.last_track_bbox = b
        self.last_bbox = b
        return True

    def track_update(self, frame):
        if not self.tracking_initialized:
            return False, None

        if self.tracker_type == "opencv":
            ok, bbox = self.tracker.update(frame)
            if not ok:
                return False, None
            b = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            return True, b

        # DaSiamRPN
        ok, b = self.dasiam.update(frame)
        if not ok:
            return False, None
        return True, b

    # -------- YOLO detection (optionally on ROI) --------
    def yolo_detect(self, frame, roi_xywh=None):
        H, W = frame.shape[:2]
        if roi_xywh is not None:
            x, y, w, h = roi_xywh
            roi = crop_roi(frame, roi_xywh)
            results = self.detector.predict(roi, conf=self.cfg.conf_threshold, verbose=False)
        else:
            results = self.detector.predict(frame, conf=self.cfg.conf_threshold, verbose=False)

        dets = []
        confs = []
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy
                b = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                if roi_xywh is not None:
                    b = map_bbox_from_roi(b, roi_xywh)
                # Clip
                b = clip_bbox_to_frame(b, W, H)
                dets.append(b)
                confs.append(conf)

        return dets, confs

    def _make_roi(self, pred_bbox, frame_shape):
        """Create ROI around pred_bbox scaled by roi_scale."""
        H, W = frame_shape[:2]
        x, y, w, h = pred_bbox
        cx, cy = center_of_xywh(pred_bbox)
        scale = max(1.0, float(self.cfg.roi_scale))
        rw = int(round(w * scale))
        rh = int(round(h * scale))
        rx = int(round(cx - rw / 2.0))
        ry = int(round(cy - rh / 2.0))
        roi = clip_bbox_to_frame([rx, ry, rw, rh], W, H)
        return roi

    # -------- Gating / selection --------
    def gate_and_select(self, dets, confs, pred_bbox):
        """Select best detection consistent with prediction. Returns (bbox, conf, ok)."""
        if not dets:
            return None, 0.0, False

        # If no prediction, pick highest conf
        if pred_bbox is None:
            i = int(np.argmax(confs))
            return dets[i], float(confs[i]), True

        pcx, pcy = center_of_xywh(pred_bbox)

        best_score = -1e9
        best = None
        best_conf = 0.0
        for b, c in zip(dets, confs):
            cx, cy = center_of_xywh(b)
            dist = math.hypot(cx - pcx, cy - pcy)
            iou = iou_xywh(b, pred_bbox)

            # hard gates
            if dist > self.cfg.gate_center_px:
                continue
            if iou < self.cfg.gate_iou_min:
                continue

            # score: prefer higher conf + higher iou - small dist penalty
            score = (2.0 * c) + (3.0 * iou) - (0.002 * dist)
            if score > best_score:
                best_score = score
                best = b
                best_conf = float(c)

        if best is None:
            return None, 0.0, False
        return best, best_conf, True

    # -------- Manual init snap --------
    def snap_manual_bbox_with_yolo(self, frame, manual_bbox_xywh):
        """Run YOLO on full frame and snap to det bbox with best IoU to manual bbox if IoU >= snap_iou_min."""
        dets, confs = self.yolo_detect(frame, roi_xywh=None)
        if not dets:
            return manual_bbox_xywh, 0.0, False
        best_b, best_i = best_overlap(dets, manual_bbox_xywh)
        if best_b is None:
            return manual_bbox_xywh, 0.0, False
        if best_i >= self.cfg.snap_iou_min:
            # accept snap
            i = dets.index(best_b)
            return best_b, float(confs[i]), True
        return manual_bbox_xywh, 0.0, False

    # -------- Main step --------
    def process_frame(self, frame):
        self.frame_idx += 1
        H, W = frame.shape[:2]

        # 1) Tracker update first
        trk_ok, trk_bbox = self.track_update(frame)
        if trk_ok:
            trk_bbox = clip_bbox_to_frame(expand_min_wh(trk_bbox, self.cfg.min_wh), W, H)
            self.last_track_bbox = trk_bbox
            self.last_bbox = trk_bbox
        else:
            if self.tracking_initialized:
                # tracker failed this frame
                self.tracking_initialized = False
                self.tracker = None
            self.last_track_bbox = None

        # 2) KF predict (if enabled) -> pred_bbox center used for gating
        pred_bbox = self.last_bbox
        if self.kf is not None:
            if self.kf.get_center() is not None:
                self.kf.predict(self.cfg.kf_dt)
                pc = self.kf.get_center()
                if pc is not None and pred_bbox is not None:
                    # replace center of pred_bbox with KF center (keep size)
                    cx, cy = pc
                    x, y, w, h = pred_bbox
                    pred_bbox = clip_bbox_to_frame(
                        [int(round(cx - w/2.0)), int(round(cy - h/2.0)), int(w), int(h)],
                        W, H
                    )

        # 3) Decide whether to run YOLO this frame
        run_det = False
        if self.tracking_initialized:
            if self.cfg.det_period > 0 and (self.frame_idx % self.cfg.det_period == 0):
                run_det = True
        else:
            if self.cfg.det_on_lost:
                run_det = True

        det_bbox = None
        det_conf = 0.0

        if run_det:
            # run YOLO on ROI if we have prediction, else full frame
            roi = None
            if pred_bbox is not None and self.tracking_initialized:
                roi = self._make_roi(pred_bbox, frame.shape)
            dets, confs = self.yolo_detect(frame, roi_xywh=roi)

            # choose a detection consistent with prediction
            cand, cconf, ok = self.gate_and_select(dets, confs, pred_bbox if pred_bbox is not None else self.last_bbox)
            if ok:
                det_bbox = cand
                det_conf = cconf

        # 4) Fuse / correct
        status = "LOST"
        out_bbox = None
        out_conf = 0.0

        if self.tracking_initialized and self.last_track_bbox is not None:
            # default output from tracker
            out_bbox = self.last_track_bbox
            status = "TRACKED"

            # if we have a valid detection correction, maybe reinit / or just update KF
            if det_bbox is not None:
                # update KF with detection center (strong correction)
                if self.kf is not None:
                    dcx, dcy = center_of_xywh(det_bbox)
                    self.kf.update(dcx, dcy)

                # reinit decision (only at correction frames)
                do_reinit = False
                if self.cfg.reinit_interval > 0 and (self.frame_idx % self.cfg.reinit_interval == 0):
                    i = iou_xywh(det_bbox, self.last_track_bbox)
                    if i < self.cfg.reinit_iou:
                        do_reinit = True

                if do_reinit:
                    self.init_tracker(frame, det_bbox)
                    out_bbox = det_bbox
                    status = "DETECTED"
                    out_conf = det_conf
                else:
                    # keep tracker bbox, but remember detection
                    out_conf = 0.0

        else:
            # not tracking right now
            if det_bbox is not None:
                self.init_tracker(frame, det_bbox)
                out_bbox = det_bbox
                status = "DETECTED"
                out_conf = det_conf
            else:
                out_bbox = None
                status = "LOST"
                out_conf = 0.0

        self.last_det_bbox = det_bbox
        self.last_conf = out_conf
        self.last_bbox = out_bbox

        return out_bbox, status, out_conf, det_bbox

# -----------------------
# UI / Drawing
# -----------------------
def draw_bbox(frame, bbox, status, conf=0.0, color=None, thickness=2):
    if bbox is None:
        return frame
    x, y, w, h = bbox
    if color is None:
        colors = {
            "DETECTED": (0, 255, 0),
            "TRACKED": (255, 165, 0),
            "LOST": (0, 0, 255),
        }
        color = colors.get(status, (255, 255, 255))
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
    label = status + (f" {conf:.2f}" if conf > 0 else "")
    cv2.putText(frame, label, (x, max(0, y-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def draw_info(frame, text, y=30):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return frame

# -----------------------
# Manual init helper
# -----------------------
def manual_init_select_bbox(cap: cv2.VideoCapture):
    """
    Manual init screen:
    - n: next frame
    - s: select ROI (cv2.selectROI)
    - q: cancel
    Returns: (bbox_xywh or None, start_frame_index)
    """
    start_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        ok, frame = cap.read()
        if not ok:
            return None, start_idx
        view = frame.copy()
        draw_info(view, f"Manual init | frame={start_idx} | keys: n(next) s(select) q(cancel)", 30)
        cv2.imshow("Manual Init", view)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('n'):
            start_idx += 1
            continue
        if k == ord('q') or k == 27:
            cv2.destroyWindow("Manual Init")
            return None, start_idx
        if k == ord('s'):
            roi = cv2.selectROI("Manual Init", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Manual Init")
            x, y, w, h = roi
            if w <= 0 or h <= 0:
                return None, start_idx
            return [int(x), int(y), int(w), int(h)], start_idx

# -----------------------
# Video loop
# -----------------------
def test_on_video(detector_tracker: DroneDetectorTracker, video_path: str, output_path: str = None, manual_init: bool = False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오 열기 실패: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 비디오: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, max(1.0, fps), (width, height))

    # manual init (optional)
    start_frame = 0
    if manual_init:
        bbox, start_frame = manual_init_select_bbox(cap)
        if bbox is not None:
            # Snap with YOLO to avoid background selection
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ok, frame0 = cap.read()
            if ok:
                snapped, sconf, snapped_ok = detector_tracker.snap_manual_bbox_with_yolo(frame0, bbox)
                if snapped_ok:
                    print(f"🧲 manual ROI -> YOLO snap ok (IoU>= {detector_tracker.cfg.snap_iou_min}). conf={sconf:.2f}")
                    bbox = snapped
                else:
                    print("⚠️ YOLO snap 실패/미통과 -> manual ROI 그대로 사용")
                detector_tracker.init_tracker(frame0, bbox)
            else:
                print("⚠️ 수동 init 프레임 읽기 실패 -> 자동 시작")
        else:
            print("⚠️ 수동 bbox 선택 취소/무효 -> 자동(YOLO) 시작")
        # Set next frame after manual init frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + 1)

    paused = False
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            bbox, status, conf, det_bbox = detector_tracker.process_frame(frame)

            # draw tracker bbox
            frame = draw_bbox(frame, bbox, status, conf)

            # draw det bbox (debug, light gray)
            if det_bbox is not None and detector_tracker.cfg.debug:
                frame = draw_bbox(frame, det_bbox, "DETECTED", 0.0, color=(200, 200, 200), thickness=1)

            frame = draw_info(frame, f"frame {frame_idx}/{total_frames} | {status} | det_period={detector_tracker.cfg.det_period} | keys: r(reinit) p(pause) ESC(exit)", 30)

            cv2.imshow("Drone Detection & Tracking (MK46)", frame)
            if writer:
                writer.write(frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        if k == ord('p'):
            paused = not paused
        if k == ord('r'):
            # pause and reselect ROI at current frame
            paused = True
            # seek back one frame to show correctly
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - 1))
            ok2, f2 = cap.read()
            if not ok2:
                paused = False
                continue
            roi = cv2.selectROI("Reinit ROI", f2, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Reinit ROI")
            x, y, w, h = roi
            if w > 0 and h > 0:
                manual_bbox = [int(x), int(y), int(w), int(h)]
                snapped, sconf, snapped_ok = detector_tracker.snap_manual_bbox_with_yolo(f2, manual_bbox)
                if snapped_ok:
                    print(f"🧲 reinit ROI -> YOLO snap ok. conf={sconf:.2f}")
                    manual_bbox = snapped
                else:
                    print("⚠️ reinit YOLO snap 실패/미통과 -> ROI 그대로 사용")
                detector_tracker.init_tracker(f2, manual_bbox)
            else:
                print("⚠️ reinit ROI 취소/무효")
            # resume
            paused = False

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("✅ 완료")

def test_on_webcam(detector_tracker: DroneDetectorTracker, manual_init: bool = False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    print("📹 웹캠 시작 (ESC 종료, r 재선택, p pause)")
    if manual_init:
        ok, frame0 = cap.read()
        if ok:
            roi = cv2.selectROI("Manual Init", frame0, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Manual Init")
            x,y,w,h = roi
            if w>0 and h>0:
                bbox = [int(x),int(y),int(w),int(h)]
                snapped, sconf, snapped_ok = detector_tracker.snap_manual_bbox_with_yolo(frame0, bbox)
                if snapped_ok:
                    print(f"🧲 manual ROI -> YOLO snap ok. conf={sconf:.2f}")
                    bbox = snapped
                detector_tracker.init_tracker(frame0, bbox)

    paused = False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            bbox, status, conf, det_bbox = detector_tracker.process_frame(frame)
            frame = draw_bbox(frame, bbox, status, conf)
            frame = draw_info(frame, f"{status} | keys: r(reinit) p(pause) ESC(exit)", 30)
            cv2.imshow("Drone Detection & Tracking (MK46)", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord('p'):
            paused = not paused
        if k == ord('r'):
            paused = True
            ok2, f2 = cap.read()
            if not ok2:
                paused = False
                continue
            roi = cv2.selectROI("Reinit ROI", f2, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Reinit ROI")
            x, y, w, h = roi
            if w > 0 and h > 0:
                manual_bbox = [int(x), int(y), int(w), int(h)]
                snapped, sconf, snapped_ok = detector_tracker.snap_manual_bbox_with_yolo(f2, manual_bbox)
                if snapped_ok:
                    print(f"🧲 reinit ROI -> YOLO snap ok. conf={sconf:.2f}")
                    manual_bbox = snapped
                detector_tracker.init_tracker(f2, manual_bbox)
            paused = False

    cap.release()
    cv2.destroyAllWindows()

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="MK46: tracking-first + periodic YOLO correction + optional KF gating")
    parser.add_argument("--model", type=str, required=True, help="YOLO model path (best.pt)")
    parser.add_argument("--video", type=str, default=None, help="video path")
    parser.add_argument("--webcam", action="store_true", help="use webcam")
    parser.add_argument("--output", type=str, default=None, help="output video path (optional)")
    parser.add_argument("--tracker", type=str, default="opencv", choices=["opencv", "dasiamrpn"], help="tracker type")
    parser.add_argument("--dasiam_model", type=str, default=None, help="DaSiamRPN model path (SiamRPNVOT.model)")

    # tracking-first knobs
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO conf threshold")
    parser.add_argument("--det_period", type=int, default=15, help="run YOLO every N frames while tracking (0=never)")
    parser.add_argument("--roi_scale", type=float, default=3.0, help="YOLO ROI scale around prediction")
    parser.add_argument("--det_on_lost", action="store_true", help="run YOLO when lost (default on)")
    parser.add_argument("--no_det_on_lost", action="store_true", help="disable YOLO when lost (not recommended)")

    # gating knobs
    parser.add_argument("--gate_center", type=float, default=140.0, help="max center distance for accepting YOLO correction")
    parser.add_argument("--gate_iou", type=float, default=0.02, help="min IoU with prediction for accepting YOLO correction")

    # init/reinit
    parser.add_argument("--manual_init", action="store_true", help="manual ROI init (n skip, s select, q cancel)")
    parser.add_argument("--snap_iou_min", type=float, default=0.05, help="manual ROI -> YOLO snap minimum IoU")
    parser.add_argument("--min_wh", type=int, default=16, help="minimum bbox w/h used for tracker init/update")
    parser.add_argument("--reinit_interval", type=int, default=10, help="check reinit each N frames (0=never)")
    parser.add_argument("--reinit_iou", type=float, default=0.30, help="reinit if IoU(det,track) < this at check time")

    # Kalman
    parser.add_argument("--no_kf", action="store_true", help="disable Kalman filter")
    parser.add_argument("--kf_dt", type=float, default=1.0, help="KF dt per frame")

    parser.add_argument("--debug", action="store_true", help="draw YOLO correction bbox (gray) for debugging")

    args = parser.parse_args()

    det_on_lost = True
    if args.no_det_on_lost:
        det_on_lost = False

    cfg = TrackConfig(
        conf_threshold=args.conf,
        det_period=args.det_period,
        det_on_lost=det_on_lost,
        roi_scale=args.roi_scale,
        snap_iou_min=args.snap_iou_min,
        gate_center_px=args.gate_center,
        gate_iou_min=args.gate_iou,
        min_wh=args.min_wh,
        reinit_interval=args.reinit_interval,
        reinit_iou=args.reinit_iou,
        use_kf=(not args.no_kf),
        kf_dt=args.kf_dt,
        debug=args.debug
    )

    detector_tracker = DroneDetectorTracker(
        detector_path=args.model,
        tracker_type=args.tracker,
        cfg=cfg,
        dasiam_model_path=args.dasiam_model
    )

    if args.webcam:
        test_on_webcam(detector_tracker, manual_init=args.manual_init)
    elif args.video:
        test_on_video(detector_tracker, args.video, args.output, manual_init=args.manual_init)
    else:
        print("❌ --video 또는 --webcam 중 하나를 선택하세요")

if __name__ == "__main__":
    main()
