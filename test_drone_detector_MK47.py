#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MK47: YOLO + Tracker (CSRT/DaSiamRPN) with "consistent detection selection" + (optional) Kalman gating.

핵심 아이디어 (추천 방식):
- YOLO는 매 프레임(또는 원하는 주기) 돌려도 되지만,
  "conf 최고 1개"를 바로 채택하지 않는다.
- 이전 상태(Tracker bbox 또는 KF 예측)와 시간적으로 일관된 detection만 선택한다.
- Tracker는 연속성을 담당하고, YOLO는 drift/occlusion 복구 및 교정을 담당한다.
- (옵션) SORT 스타일의 간단 KF(중심+속도)로 예측을 만들고 게이팅에 사용한다.

기능:
- --manual_init: 시작 시 프레임 스킵(n) + ROI 선택(s) + 취소(q)
- OpenCV 4.12 bbox 타입 문제 회피(legacy tracker + int 강제 변환)
- "갑자기 다른 물체로 갈아타는" 오탐 억제를 위해
  IoU/center-distance 게이팅 + scoring + hysteresis(연속 프레임 확인) 적용
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------
# Optional DaSiamRPN import
# -------------------------
try:
    from net import SiamRPNvot
    from run_SiamRPN import SiamRPN_init, SiamRPN_track
    from utils import get_axis_aligned_bbox, cxy_wh_2_rect
    DASIAM_IMPORT_ERROR = None
except Exception as e:
    SiamRPNvot = None
    SiamRPN_init = None
    SiamRPN_track = None
    get_axis_aligned_bbox = None
    cxy_wh_2_rect = None
    DASIAM_IMPORT_ERROR = e


# -------------------------
# Small utils
# -------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    x = int(round(x1))
    y = int(round(y1))
    w = int(round(x2 - x1))
    h = int(round(y2 - y1))
    return [x, y, w, h]

def xywh_to_cxcywh(b):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0, w, h)

def cxcywh_to_xywh(cx, cy, w, h):
    return [int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h))]

def iou_xywh(a, b):
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0

def center_dist(a, b):
    if a is None or b is None:
        return 1e9
    acx, acy, _, _ = xywh_to_cxcywh(a)
    bcx, bcy, _, _ = xywh_to_cxcywh(b)
    return float(math.hypot(acx - bcx, acy - bcy))

def expand_bbox_keep_center(b, W, H, min_wh=24):
    """너무 작은 bbox는 tracking 안정성을 위해 최소 크기 확보(중심 유지)."""
    x, y, w, h = b
    cx, cy, _, _ = xywh_to_cxcywh(b)
    w2 = max(w, min_wh)
    h2 = max(h, min_wh)
    x2 = int(round(cx - w2 / 2.0))
    y2 = int(round(cy - h2 / 2.0))
    x2 = clamp(x2, 0, W - 1)
    y2 = clamp(y2, 0, H - 1)
    w2 = clamp(w2, 1, W - x2)
    h2 = clamp(h2, 1, H - y2)
    return [int(x2), int(y2), int(w2), int(h2)]


# -------------------------
# SORT-style tiny KF (center + velocity)
# -------------------------
@dataclass
class KFParams:
    q: float = 3.0     # process noise
    r: float = 25.0    # measurement noise (pixels^2-ish)
    p0: float = 500.0  # initial covariance

class CenterKF:
    """
    Constant-velocity Kalman filter on (cx, cy, vx, vy).
    Measurement: (cx, cy).
    This is "SORT-like" but simplified (no aspect ratio / scale state).
    """
    def __init__(self, params: KFParams):
        self.params = params
        self.initialized = False
        self.x = np.zeros((4, 1), dtype=np.float32)  # [cx, cy, vx, vy]
        self.P = np.eye(4, dtype=np.float32) * params.p0

        # H maps state -> measurement
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * params.r

    def init(self, cx, cy):
        self.x[:] = 0
        self.x[0, 0] = float(cx)
        self.x[1, 0] = float(cy)
        self.P = np.eye(4, dtype=np.float32) * self.params.p0
        self.initialized = True

    def predict(self, dt=1.0):
        if not self.initialized:
            return None
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        # Process noise (very simple)
        q = self.params.q
        Q = np.eye(4, dtype=np.float32) * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, cx, cy):
        if not self.initialized:
            self.init(cx, cy)
            return
        z = np.array([[float(cx)], [float(cy)]], dtype=np.float32)
        y = z - (self.H @ self.x)  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def get_center(self):
        if not self.initialized:
            return None
        return float(self.x[0, 0]), float(self.x[1, 0])


# -------------------------
# Main detector+tracker
# -------------------------
class DroneDetectorTracker:
    def __init__(
        self,
        detector_path: str,
        tracker_type: str = "opencv",
        conf_threshold: float = 0.3,
        # selection/gating
        gate_center_px: float = 120.0,
        gate_iou_min: float = 0.02,
        score_alpha: float = 1.0,   # conf weight
        score_beta: float = 2.0,    # IoU weight
        score_gamma: float = 0.005, # distance weight (px -> penalty)
        # hysteresis
        switch_iou_thresh: float = 0.2,
        switch_confirm_frames: int = 3,
        # tracker reinit
        reinit_interval: int = 10,
        reinit_iou: float = 0.4,
        min_wh: int = 24,
        # kalman
        use_kf: bool = True,
        kf_q: float = 3.0,
        kf_r: float = 25.0,
        # dasiam
        dasiam_model_path: str = None,
        debug: bool = False,
    ):
        print(f"🔍 YOLO 로드: {detector_path}")
        self.detector = YOLO(detector_path)
        self.conf_threshold = conf_threshold

        self.tracker_type = tracker_type
        self.tracker = None
        self.tracking_initialized = False

        self.reinit_interval = reinit_interval
        self.reinit_iou = reinit_iou
        self.last_init_frame = -10**9
        self.frame_idx = 0
        self.min_wh = int(min_wh)

        # gating/scoring
        self.gate_center_px = float(gate_center_px)
        self.gate_iou_min = float(gate_iou_min)
        self.score_alpha = float(score_alpha)
        self.score_beta = float(score_beta)
        self.score_gamma = float(score_gamma)

        # hysteresis
        self.switch_iou_thresh = float(switch_iou_thresh)
        self.switch_confirm_frames = int(switch_confirm_frames)
        self._pending_bbox = None
        self._pending_count = 0

        self.last_bbox = None

        # KF
        self.use_kf = bool(use_kf)
        self.kf = CenterKF(KFParams(q=float(kf_q), r=float(kf_r))) if self.use_kf else None

        # DaSiamRPN
        self.dasiam_model_path = dasiam_model_path
        self.net = None
        self.state = None
        if tracker_type == "dasiamrpn":
            self._init_dasiamrpn_net(dasiam_model_path)

        self.debug = debug
        print(f"✅ 초기화 완료 (tracker={tracker_type}, use_kf={self.use_kf})")

    # ---------- detection ----------
    def detect_all(self, frame):
        """
        Return all detections (xywh) and confs.
        """
        results = self.detector.predict(frame, conf=self.conf_threshold, verbose=False)
        if len(results) == 0:
            return [], []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return [], []
        confs = boxes.conf.detach().cpu().numpy().astype(np.float32).tolist()
        xyxys = boxes.xyxy.detach().cpu().numpy()
        dets = [xyxy_to_xywh(xyxy) for xyxy in xyxys]
        return dets, confs

    def select_consistent_detection(self, dets, confs, ref_bbox, W, H):
        """
        Choose the most temporally consistent detection w.r.t. ref_bbox (tracker bbox or KF-pred bbox).
        Uses:
        - hard gate: center distance + IoU
        - score: alpha*conf + beta*iou - gamma*dist
        - hysteresis: if det jumps (IoU small), confirm for K frames
        """
        if len(dets) == 0:
            self._pending_bbox, self._pending_count = None, 0
            return None, 0.0, False

        # if no reference, fall back to max conf
        if ref_bbox is None:
            i = int(np.argmax(np.array(confs)))
            self._pending_bbox, self._pending_count = None, 0
            return dets[i], float(confs[i]), True

        best = None
        best_conf = 0.0
        best_score = -1e18

        for b, c in zip(dets, confs):
            # Clamp bbox into frame
            x, y, w, h = b
            x = clamp(x, 0, W - 1)
            y = clamp(y, 0, H - 1)
            w = clamp(w, 1, W - x)
            h = clamp(h, 1, H - y)
            b = [int(x), int(y), int(w), int(h)]

            dist = center_dist(b, ref_bbox)
            iou = iou_xywh(b, ref_bbox)

            # hard gates
            if dist > self.gate_center_px:
                continue
            if iou < self.gate_iou_min:
                continue

            score = self.score_alpha * float(c) + self.score_beta * float(iou) - self.score_gamma * float(dist)
            if score > best_score:
                best_score = score
                best = b
                best_conf = float(c)

        if best is None:
            # nothing passed gating
            self._pending_bbox, self._pending_count = None, 0
            return None, 0.0, False

        # hysteresis: if it "jumps" away from ref (low IoU), require K frames
        jump_iou = iou_xywh(best, ref_bbox)
        if jump_iou < self.switch_iou_thresh:
            if self._pending_bbox is None or iou_xywh(best, self._pending_bbox) < 0.3:
                self._pending_bbox = best
                self._pending_count = 1
            else:
                self._pending_count += 1

            if self._pending_count >= self.switch_confirm_frames:
                self._pending_bbox, self._pending_count = None, 0
                return best, best_conf, True
            else:
                # not confirmed yet
                return None, 0.0, False

        # stable enough -> accept immediately
        self._pending_bbox, self._pending_count = None, 0
        return best, best_conf, True

    # ---------- tracker init/update ----------
    def init_tracker(self, frame, bbox_xywh):
        H, W = frame.shape[:2]
        bbox_xywh = expand_bbox_keep_center(bbox_xywh, W, H, min_wh=self.min_wh)

        if self.tracker_type == "opencv":
            # OpenCV 4.12: legacy tracker is often more consistent on some builds
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                self.tracker = cv2.legacy.TrackerCSRT_create()
            else:
                self.tracker = cv2.TrackerCSRT_create()

            # IMPORTANT: pass ints
            x, y, w, h = bbox_xywh
            x, y, w, h = int(x), int(y), int(w), int(h)
            self.tracker.init(frame, (x, y, w, h))
            self.tracking_initialized = True
            print(f"📍 CSRT 추적기 초기화: {bbox_xywh}")

        elif self.tracker_type == "dasiamrpn":
            if self.net is None:
                raise RuntimeError("DaSiamRPN net not initialized.")
            # Convert bbox_xywh to target_pos/target_sz for SiamRPN_init
            cx, cy, w, h = xywh_to_cxcywh(bbox_xywh)
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            self.state = SiamRPN_init(frame, target_pos, target_sz, self.net)
            self.tracking_initialized = True
            print("📍 DaSiamRPN 추적기 초기화")

        else:
            raise ValueError(f"Unknown tracker_type: {self.tracker_type}")

        self.last_bbox = bbox_xywh
        # KF init/update
        if self.use_kf:
            cx, cy, _, _ = xywh_to_cxcywh(bbox_xywh)
            if not self.kf.initialized:
                self.kf.init(cx, cy)
            else:
                self.kf.update(cx, cy)

    def track(self, frame):
        if not self.tracking_initialized:
            return None

        if self.tracker_type == "opencv":
            ok, bbox = self.tracker.update(frame)
            if not ok:
                self.tracking_initialized = False
                return None
            x, y, w, h = bbox
            return [int(x), int(y), int(w), int(h)]

        if self.tracker_type == "dasiamrpn":
            if self.state is None:
                self.tracking_initialized = False
                return None
            self.state = SiamRPN_track(self.state, frame)
            res = cxy_wh_2_rect(self.state["target_pos"], self.state["target_sz"])
            res = [int(v) for v in res]
            return res

        return None

    # ---------- main logic ----------
    def process_frame(self, frame):
        self.frame_idx += 1
        H, W = frame.shape[:2]

        # 1) Tracker update first (for continuity)
        trk_bbox = None
        if self.tracking_initialized:
            trk_bbox = self.track(frame)

        # 2) KF predict (SORT-style) for gating reference
        pred_bbox = None
        if self.use_kf and self.kf.initialized:
            pc = self.kf.predict(dt=1.0)
            if pc is not None:
                pcx, pcy = pc
                # use last size (from last_bbox or tracker bbox) to form a predicted bbox
                size_src = trk_bbox if trk_bbox is not None else self.last_bbox
                if size_src is not None:
                    _, _, pw, ph = xywh_to_cxcywh(size_src)
                    pred_bbox = expand_bbox_keep_center(cxcywh_to_xywh(pcx, pcy, pw, ph), W, H, min_wh=self.min_wh)

        # reference bbox priority: tracker -> kf pred -> last_bbox
        ref_bbox = trk_bbox if trk_bbox is not None else (pred_bbox if pred_bbox is not None else self.last_bbox)

        # 3) YOLO detect all (full-frame; if you want speed, later we can add ROI mode)
        dets, confs = self.detect_all(frame)

        # 4) Select consistent detection against ref
        det_bbox, det_conf, det_ok = self.select_consistent_detection(dets, confs, ref_bbox, W, H)

        if self.debug and len(dets) > 0:
            msg = f"[dbg] dets={len(dets)} trk={'Y' if trk_bbox else 'N'} ref={'Y' if ref_bbox else 'N'} sel={'Y' if det_ok else 'N'}"
            print(msg)

        # 5) If we have a valid detection, decide whether to re-init tracker
        if det_ok and det_bbox is not None:
            init_bbox = expand_bbox_keep_center(det_bbox, W, H, min_wh=self.min_wh)

            should_reinit = False
            if (not self.tracking_initialized) or (trk_bbox is None):
                should_reinit = True
            else:
                # conditionally reinit to correct drift
                if (self.frame_idx - self.last_init_frame) >= self.reinit_interval:
                    if iou_xywh(init_bbox, trk_bbox) < self.reinit_iou:
                        should_reinit = True

            if should_reinit:
                self.init_tracker(frame, init_bbox)
                self.last_init_frame = self.frame_idx
                return det_bbox, "DETECTED", float(det_conf), det_bbox

            # tracker alive: still update KF with detection center (measurement) to stabilize gating
            if self.use_kf:
                cx, cy, _, _ = xywh_to_cxcywh(det_bbox)
                self.kf.update(cx, cy)

        # 6) If tracker result exists, use it
        if trk_bbox is not None:
            self.last_bbox = trk_bbox
            if self.use_kf:
                cx, cy, _, _ = xywh_to_cxcywh(trk_bbox)
                self.kf.update(cx, cy)
            return trk_bbox, "TRACKED", 0.0, det_bbox if det_ok else None

        # 7) If no tracker but we have a detection (even if not ok), optionally fall back to max-conf
        if (not self.tracking_initialized) and len(dets) > 0:
            # conservative: only if no ref exists
            if ref_bbox is None:
                i = int(np.argmax(np.array(confs)))
                b = dets[i]
                self.init_tracker(frame, b)
                self.last_init_frame = self.frame_idx
                return b, "DETECTED", float(confs[i]), b

        return None, "LOST", 0.0, det_bbox if det_ok else None

    # ---------- DaSiamRPN init ----------
    def _init_dasiamrpn_net(self, model_path):
        if SiamRPNvot is None or SiamRPN_init is None or SiamRPN_track is None:
            print("❌ DaSiamRPN import 실패:", DASIAM_IMPORT_ERROR)
            raise RuntimeError("DaSiamRPN 모듈 import가 안됩니다. (net/run_SiamRPN.py/utils.py 확인)")
        if model_path is None:
            raise ValueError("--dasiam_model 경로를 지정해야 합니다.")
        model_path = str(model_path)

        self.net = SiamRPNvot()
        self.net.load_state_dict(torch_load_compat(model_path))
        self.net.eval().cuda()
        print(f"✅ DaSiamRPN 모델 로드: {model_path}")

def torch_load_compat(path):
    # Local import to avoid torch dependency when using only opencv tracker.
    import torch
    # map_location not needed if you have CUDA; keep simple.
    return torch.load(path)


# -------------------------
# Visualization
# -------------------------
def draw_bbox(frame, bbox, status, conf=0.0, det_bbox=None):
    if bbox is not None:
        x, y, w, h = bbox
        colors = {"DETECTED": (0, 255, 0), "TRACKED": (255, 165, 0), "LOST": (0, 0, 255)}
        color = colors.get(status, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = status + (f" {conf:.2f}" if conf > 0 else "")
        cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # optional: show selected detection bbox (gray)
    if det_bbox is not None:
        x, y, w, h = det_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)

    return frame


# -------------------------
# Manual init helper
# -------------------------
def manual_init_select_bbox(cap):
    """
    Manual init UI:
    - shows current frame
    - 'n': next frame
    - 's': select ROI on current frame
    - 'q'/'ESC': cancel manual init
    Returns: (frame, bbox_xywh) or (None, None) if cancelled
    """
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            return None, None
        idx += 1

        disp = frame.copy()
        cv2.putText(disp, f"Manual Init | frame={idx} | 'n':next  's':select ROI  'q':cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Manual Init", disp)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("n"):
            continue
        if key == ord("s"):
            bbox = cv2.selectROI("Manual Init", frame, fromCenter=False, showCrosshair=True)
            x, y, w, h = [int(v) for v in bbox]
            if w > 0 and h > 0:
                cv2.destroyWindow("Manual Init")
                return frame, [x, y, w, h]
            else:
                print("⚠️ ROI 무효. 다시 선택하세요.")
                continue
        if key == ord("q") or key == 27:
            cv2.destroyWindow("Manual Init")
            return None, None


# -------------------------
# Video test
# -------------------------
def test_on_video(detector_tracker, video_path, output_path=None, manual_init=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 비디오 열기 실패: {video_path}")
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 비디오: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # manual init (optional)
    if manual_init:
        frame0, bbox0 = manual_init_select_bbox(cap)
        if frame0 is not None and bbox0 is not None:
            detector_tracker.init_tracker(frame0, bbox0)
        else:
            print("⚠️ 수동 bbox 선택 취소/무효 -> 자동 시작")

    frame_idx = 0
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            bbox, status, conf, det_bbox = detector_tracker.process_frame(frame)
            vis = draw_bbox(frame, bbox, status, conf, det_bbox=det_bbox)
            cv2.putText(vis, f"Frame {frame_idx}/{total_frames} | {status}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("YOLO + Tracker (MK47)", vis)
            if out is not None:
                out.write(vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("r"):
            # re-init manually at current position (pause first)
            paused = True
            # show current frame and select ROI
            bbox_new = cv2.selectROI("YOLO + Tracker (MK47)", frame, fromCenter=False, showCrosshair=True)
            x, y, w, h = [int(v) for v in bbox_new]
            if w > 0 and h > 0:
                detector_tracker.init_tracker(frame, [x, y, w, h])
                detector_tracker.last_init_frame = detector_tracker.frame_idx
            paused = False

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"✅ 완료: {frame_idx} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str, help="YOLO best.pt path")
    ap.add_argument("--video", type=str, default=None, help="video path")
    ap.add_argument("--output", type=str, default=None, help="save output mp4")
    ap.add_argument("--tracker", type=str, default="opencv", choices=["opencv", "dasiamrpn"])
    ap.add_argument("--conf", type=float, default=0.3)

    # gating/scoring
    ap.add_argument("--gate_center", type=float, default=120.0)
    ap.add_argument("--gate_iou", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=1.0, help="conf weight")
    ap.add_argument("--beta", type=float, default=2.0, help="IoU weight")
    ap.add_argument("--gamma", type=float, default=0.005, help="distance penalty weight")

    # hysteresis
    ap.add_argument("--switch_iou", type=float, default=0.2)
    ap.add_argument("--switch_k", type=int, default=3)

    # reinit
    ap.add_argument("--reinit_interval", type=int, default=10)
    ap.add_argument("--reinit_iou", type=float, default=0.4)
    ap.add_argument("--min_wh", type=int, default=24)

    # kalman
    ap.add_argument("--use_kf", action="store_true", help="enable simple KF gating (SORT-like)")
    ap.add_argument("--kf_q", type=float, default=3.0)
    ap.add_argument("--kf_r", type=float, default=25.0)

    # manual init / dasiam
    ap.add_argument("--manual_init", action="store_true")
    ap.add_argument("--dasiam_model", type=str, default=None, help="SiamRPNVOT.model path")

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.video is None:
        print("❌ --video 필요")
        return

    dt = DroneDetectorTracker(
        detector_path=args.model,
        tracker_type=args.tracker,
        conf_threshold=args.conf,
        gate_center_px=args.gate_center,
        gate_iou_min=args.gate_iou,
        score_alpha=args.alpha,
        score_beta=args.beta,
        score_gamma=args.gamma,
        switch_iou_thresh=args.switch_iou,
        switch_confirm_frames=args.switch_k,
        reinit_interval=args.reinit_interval,
        reinit_iou=args.reinit_iou,
        min_wh=args.min_wh,
        use_kf=args.use_kf,
        kf_q=args.kf_q,
        kf_r=args.kf_r,
        dasiam_model_path=args.dasiam_model,
        debug=args.debug,
    )

    test_on_video(dt, args.video, args.output, manual_init=args.manual_init)


if __name__ == "__main__":
    main()
