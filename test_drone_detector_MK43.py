#!/usr/bin/env python3
"""
YOLO + Tracker 통합 테스트 (D-bias 스타일)
- Detector 우선 (YOLO)
- Detector 실패 시 Tracker 사용
- Detector가 매 프레임 작은 bbox를 내놓을 때도 "작은 물체 추적"이 되도록:
  1) 탐지될 때마다 매번 re-init 하지 않고, 필요할 때만 re-init (IoU + 최소 간격)
  2) bbox가 아주 작더라도 tracker init은 "센터 유지 + 컨텍스트 확보(min_wh 확장)"로 안정화
- Tracker: opencv(CSRT) 또는 DaSiamRPN

필요:
- ultralytics
- opencv-python
- torch
- DaSiamRPN 코드(현재 폴더/패키지로 import 가능해야 함):
    net.py (SiamRPNvot)
    run_SiamRPN.py (SiamRPN_init, SiamRPN_track)
    utils.py (optional)
    SiamRPNVOT.model (가중치 파일)

사용 예:
  python3 test_drone_detector_MK42_modified.py --model best.pt --video input.mp4 --tracker dasiamrpn --dasiam_model SiamRPNVOT.model --manual_init
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time

from typing import Optional

# -----------------------------
# DaSiamRPN imports
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
    - 탐지 결과가 자주 나오더라도, tracker re-init은 "필요할 때만"
    """

    def __init__(
        self,
        detector_path: str,
        tracker_type: str = "opencv",
        conf_threshold: float = 0.5,
        
        #dasiam_model_path: str | None = None,
        dasiam_model_path: Optional[str] = None,
        dasiam_use_cuda: bool = True,



        # --- stability knobs ---
        small_object_min_wh: int = 32,
        reinit_iou_thresh: float = 0.2,
        reinit_min_interval: int = 10,
        verbose_reinit: bool = False,
    ):
        """
        Args:
            detector_path: YOLO 모델 경로
            tracker_type: 'opencv' | 'dasiamrpn'
            conf_threshold: YOLO conf threshold
            dasiam_model_path: SiamRPNVOT.model 경로 (tracker_type='dasiamrpn'일 때 필요)
            dasiam_use_cuda: CUDA 사용 여부
            small_object_min_wh: bbox가 너무 작을 때 init 안정화를 위한 "컨텍스트 확장" 최소 크기
            reinit_iou_thresh: det(init_bbox)와 track bbox IoU가 이 값보다 작으면 re-init 고려
            reinit_min_interval: re-init 최소 프레임 간격 (너무 잦은 init 방지)
            verbose_reinit: re-init 판단 로그 출력 여부
        """
        print(f"🔍 YOLO 로드: {detector_path}")
        self.detector = YOLO(detector_path)
        self.conf_threshold = float(conf_threshold)

        self.tracker_type = tracker_type
        self.tracker = None
        self.tracking_initialized = False

        # DaSiamRPN state/net
        self.dasiam_net = None
        self.dasiam_state = None
        self.dasiam_use_cuda = dasiam_use_cuda

        # stability knobs
        self.small_object_min_wh = int(small_object_min_wh)
        self.reinit_iou_thresh = float(reinit_iou_thresh)
        self.reinit_min_interval = int(reinit_min_interval)
        self.verbose_reinit = bool(verbose_reinit)

        self.frame_idx = 0
        self.last_init_frame = -10**9

        if self.tracker_type == "dasiamrpn":
            self._init_dasiamrpn_net(dasiam_model_path)

        print(f"✅ 초기화 완료 (추적기: {tracker_type})")

    # -----------------------------
    # Utils
    # -----------------------------
    @staticmethod
    def _clamp_bbox_xywh(bbox, W, H):
        x, y, w, h = bbox
        x = int(x); y = int(y); w = int(w); h = int(h)
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        x2 = max(0, min(x + w, W - 1))
        y2 = max(0, min(y + h, H - 1))
        w2 = max(1, x2 - x)
        h2 = max(1, y2 - y)
        return [x, y, w2, h2]

    @staticmethod
    def _iou_xywh(a, b) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return float(inter / union) if union > 0 else 0.0

    def _expand_bbox_keep_center(self, bbox, W, H, min_wh: int):
        """
        bbox가 아주 작더라도 추적 init을 안정화시키기 위해:
        - 중심(cx,cy)은 유지
        - init_bbox는 최소 min_wh x min_wh 이상이 되도록 확장(컨텍스트 확보)
        """
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        new_w = max(int(w), int(min_wh))
        new_h = max(int(h), int(min_wh))

        nx = int(round(cx - new_w / 2.0))
        ny = int(round(cy - new_h / 2.0))

        # clamp
        nx = max(0, min(nx, W - 1))
        ny = max(0, min(ny, H - 1))

        nx2 = min(W - 1, nx + new_w)
        ny2 = min(H - 1, ny + new_h)

        return [nx, ny, max(1, nx2 - nx), max(1, ny2 - ny)]

    # -----------------------------
    # Tracker init
    # -----------------------------
    def init_tracker(self, frame, bbox_xywh):
        """
        bbox_xywh: [x, y, w, h]
        - 여기서는 "너무 작다"로 거절하지 않음 (작은 물체도 tracking해야 하니까)
        - 대신 init_bbox는 process_frame에서 컨텍스트 확보로 확장해서 들어오게 함
        """
        if bbox_xywh is None:
            self.tracking_initialized = False
            return

        H, W = frame.shape[:2]
        x, y, w, h = bbox_xywh
        if w < 1 or h < 1:
            self.tracking_initialized = False
            return

        bbox_xywh = self._clamp_bbox_xywh(bbox_xywh, W, H)

        if self.tracker_type == "opencv":
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame, (float(bbox_xywh[0]), float(bbox_xywh[1]), float(bbox_xywh[2]), float(bbox_xywh[3])))
            self.tracking_initialized = True
            print("📍 OpenCV CSRT 추적기 초기화")

        elif self.tracker_type == "dasiamrpn":
            if self.dasiam_net is None:
                print("❌ DaSiamRPN net not loaded.")
                self.tracking_initialized = False
                return

            x, y, w, h = bbox_xywh
            cx = x + w / 2.0
            cy = y + h / 2.0
            target_pos = np.array([cx, cy], dtype=np.float32)
            target_sz = np.array([w, h], dtype=np.float32)

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

        if self.tracker_type == "opencv":
            if self.tracker is None:
                return None
            success, bbox = self.tracker.update(frame)
            if not success:
                self.tracking_initialized = False
                return None
            x, y, w, h = bbox
            return [int(x), int(y), int(w), int(h)]

        if self.tracker_type == "dasiamrpn":
            if self.dasiam_state is None:
                self.tracking_initialized = False
                return None

            self.dasiam_state = SiamRPN_track(self.dasiam_state, frame)

            pos = self.dasiam_state.get("target_pos", None)
            sz = self.dasiam_state.get("target_sz", None)
            if pos is None or sz is None:
                self.tracking_initialized = False
                return None

            cx, cy = float(pos[0]), float(pos[1])
            w, h = float(sz[0]), float(sz[1])

            # 여기서도 "작다"로 바로 실패 처리하지 말고, 유한성만 체크
            if not np.isfinite([cx, cy, w, h]).all() or w <= 0 or h <= 0:
                self.tracking_initialized = False
                return None

            x = int(cx - w / 2.0)
            y = int(cy - h / 2.0)

            H, W = frame.shape[:2]
            return self._clamp_bbox_xywh([x, y, int(w), int(h)], W, H)

        return None

    # -----------------------------
    # D-bias process (핵심)
    # -----------------------------
    def process_frame(self, frame):
        """
        작은 bbox도 추적 가능하도록:
        - 탐지되었다고 매 프레임 re-init 하지 않음
        - init 시에는 bbox 중심 유지 + 컨텍스트 확보(min_wh 확장)
        """
        self.frame_idx += 1
        H, W = frame.shape[:2]

        det_bbox, conf = self.detect(frame)

        # 먼저 tracking 업데이트(가능하면)
        trk_bbox = None
        if self.tracking_initialized:
            trk_bbox = self.track(frame)

        # 탐지가 있으면, "필요할 때만" re-init 판단
        if det_bbox is not None:
            det_bbox = self._clamp_bbox_xywh(det_bbox, W, H)

            init_bbox = self._expand_bbox_keep_center(
                det_bbox, W, H, min_wh=self.small_object_min_wh
            )
            init_bbox = self._clamp_bbox_xywh(init_bbox, W, H)

            should_reinit = False
            reason = ""

            if (not self.tracking_initialized) or (trk_bbox is None):
                should_reinit = True
                reason = "no tracker"
            else:
                # 너무 자주 init 방지
                if (self.frame_idx - self.last_init_frame) >= self.reinit_min_interval:
                    iou = self._iou_xywh(init_bbox, trk_bbox)
                    if iou < self.reinit_iou_thresh:
                        should_reinit = True
                        reason = f"low IoU {iou:.2f} < {self.reinit_iou_thresh:.2f}"
                    else:
                        reason = f"IoU ok {iou:.2f}"
                else:
                    reason = "cooldown"

            if self.verbose_reinit and det_bbox is not None:
                print(f"[reinit?] frame={self.frame_idx} reason={reason} det={det_bbox} init={init_bbox} trk={trk_bbox}")

            if should_reinit:
                self.init_tracker(frame, init_bbox)
                self.last_init_frame = self.frame_idx
                # 표시용 bbox는 det_bbox(작아도 OK). init_bbox로 표시하고 싶으면 init_bbox로 바꾸면 됨.
                return det_bbox, "DETECTED", conf

            # re-init 안 하면 tracking 결과가 있으면 tracking을 우선 사용(더 부드러움)
            if trk_bbox is not None:
                return trk_bbox, "TRACKED", 0.0

            # tracking이 없다면 탐지 bbox라도 반환
            return det_bbox, "DETECTED", conf

        # 탐지 없으면 tracking만
        if trk_bbox is not None:
            return trk_bbox, "TRACKED", 0.0

        return None, "LOST", 0.0

    # -----------------------------
    # DaSiamRPN net load
    # -----------------------------
    def _init_dasiamrpn_net(self, model_path):
        if SiamRPNvot is None or SiamRPN_init is None or SiamRPN_track is None:
            print("❌ DaSiamRPN import 실패:", DASIAM_IMPORT_ERROR)
            raise RuntimeError("DaSiamRPN 모듈 import가 안됩니다. (net/run_SiamRPN/utils 확인)")

        if model_path is None:
            model_path = join(realpath(dirname(__file__)), "SiamRPNVOT.model")

        model_path = str(model_path)
        print(f"🧠 DaSiamRPN 로드: {model_path}")

        self.dasiam_net = SiamRPNvot()
        state = torch.load(model_path, map_location="cpu")
        self.dasiam_net.load_state_dict(state)
        self.dasiam_net.eval()

        if self.dasiam_use_cuda and torch.cuda.is_available():
            self.dasiam_net = self.dasiam_net.cuda()
            print("🚀 DaSiamRPN: CUDA 사용")
        else:
            print("🧊 DaSiamRPN: CPU 사용 (CUDA 비활성/불가)")

        self.dasiam_state = None
        self.tracking_initialized = False


def draw_bbox(frame, bbox, status, conf=0.0):
    if bbox is None:
        return frame

    x, y, w, h = bbox

    colors = {
        "DETECTED": (0, 255, 0),
        "TRACKED": (255, 165, 0),
        "LOST": (0, 0, 255),
    }
    color = colors.get(status, (255, 255, 255))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"{status}"
    if conf > 0:
        label += f" {conf:.2f}"

    cv2.putText(
        frame,
        label,
        (x, max(0, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    return frame


def test_on_video(detector_tracker: DroneDetectorTracker, video_path, output_path=None, manual_init=False):
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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # -----------------------------
    # Manual init at start (ROI 선택)
    # -----------------------------
    if manual_init:
        ret, first = cap.read()
        if not ret:
            print("❌ 첫 프레임 읽기 실패")
            cap.release()
            return

        roi = cv2.selectROI(
            "Select Initial BBox (ENTER to confirm, ESC to cancel)",
            first,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyWindow("Select Initial BBox (ENTER to confirm, ESC to cancel)")

        x, y, w, h = roi
        if w > 0 and h > 0:
            bbox = [int(x), int(y), int(w), int(h)]
            # 작은 물체여도 init 안정화를 위해 컨텍스트 확보(min_wh 확장)
            bbox_init = detector_tracker._expand_bbox_keep_center(bbox, width, height, detector_tracker.small_object_min_wh)
            detector_tracker.init_tracker(first, bbox_init)
            detector_tracker.last_init_frame = 0
            print(f"✅ 수동 초기 bbox: {bbox} (init_bbox={bbox_init})")
        else:
            print("⚠️ 수동 bbox 선택 취소/무효 -> 자동(YOLO) 시작")

        # 첫 프레임부터 루프 재시작
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Drone Detection & Tracking", frame)
        if out is not None:
            out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"✅ 완료: {frame_idx} frames")


def test_on_webcam(detector_tracker: DroneDetectorTracker):
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
        cv2.putText(
            frame,
            f"{status} | FPS {curr_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Drone Detection & Tracking (Webcam)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLO + Tracker 통합 테스트 (DaSiamRPN 포함)")

    parser.add_argument("--model", type=str, required=True, help="YOLO 모델 경로 (best.pt)")
    parser.add_argument("--video", type=str, help="테스트 비디오 경로")
    parser.add_argument("--webcam", action="store_true", help="웹캠 사용")
    parser.add_argument("--output", type=str, help="결과 비디오 저장 경로")
    parser.add_argument("--conf", type=float, default=0.5, help="탐지 신뢰도 임계값")

    parser.add_argument(
        "--tracker",
        type=str,
        default="opencv",
        choices=["opencv", "dasiamrpn"],
        help="추적기 종류: opencv | dasiamrpn",
    )

    parser.add_argument("--dasiam_model", type=str, default=None, help="DaSiamRPN weight 경로 (SiamRPNVOT.model)")
    parser.add_argument("--dasiam_cpu", action="store_true", help="DaSiamRPN을 CPU로 강제 실행")

    # --- new knobs ---
    parser.add_argument("--manual_init", action="store_true", help="비디오 시작 시 수동 bbox 선택으로 tracker 초기화")
    parser.add_argument("--min_wh", type=int, default=32, help="작은 물체 init 안정화를 위한 최소 컨텍스트 크기 (기본 32)")
    parser.add_argument("--reinit_iou", type=float, default=0.2, help="re-init IoU 임계값 (기본 0.2)")
    parser.add_argument("--reinit_interval", type=int, default=10, help="re-init 최소 프레임 간격 (기본 10)")
    parser.add_argument("--verbose_reinit", action="store_true", help="re-init 판단 로그 출력")

    args = parser.parse_args()

    detector_tracker = DroneDetectorTracker(
        detector_path=args.model,
        tracker_type=args.tracker,
        conf_threshold=args.conf,
        dasiam_model_path=args.dasiam_model,
        dasiam_use_cuda=(not args.dasiam_cpu),
        small_object_min_wh=args.min_wh,
        reinit_iou_thresh=args.reinit_iou,
        reinit_min_interval=args.reinit_interval,
        verbose_reinit=args.verbose_reinit,
    )

    if args.webcam:
        test_on_webcam(detector_tracker)
    elif args.video:
        test_on_video(detector_tracker, args.video, args.output, manual_init=args.manual_init)
    else:
        print("❌ --video 또는 --webcam 중 하나를 선택하세요")


if __name__ == "__main__":
    main()
