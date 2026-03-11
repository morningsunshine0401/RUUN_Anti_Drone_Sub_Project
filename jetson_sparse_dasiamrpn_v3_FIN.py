#!/usr/bin/env python3
"""
Hybrid Detection + DaSiamRPN tracker

YOLO runs every frame.
DaSiamRPN runs every N frames for validation.
Cached tracking results are used in between to reduce computation.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
from collections import deque
import torch
import sys
from os.path import realpath, dirname, join

# DaSiamRPN modules
sys.path.insert(0, './DaSiamRPN/code')
from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import cxy_wh_2_rect


class SparseDaSiamRPNTracker:
    """
    Detection + tracking hybrid pipeline.
    Tracking is executed periodically to reduce runtime cost.
    """

    def __init__(self, detector_path, conf_threshold=0.5, jump_threshold=100,
                 dasiamrpn_model='./DaSiamRPN/code/SiamRPNVOT.model',
                 tracking_interval=3):

        print(f"Loading YOLO model: {detector_path}")
        self.detector = YOLO(detector_path)
        self.conf_threshold = conf_threshold
        self.jump_threshold = jump_threshold
        self.tracking_interval = tracking_interval

        # Load DaSiamRPN tracker
        print(f"Loading DaSiamRPN model: {dasiamrpn_model}")
        self.net = SiamRPNvot()
        self.net.load_state_dict(torch.load(dasiamrpn_model))
        self.net.eval()

        if torch.cuda.is_available():
            self.net = self.net.cuda()
            print("DaSiamRPN running on CUDA")
        else:
            print("DaSiamRPN running on CPU")

        self.dasiamrpn_state = None
        self.tracking_initialized = False
        self.last_track_bbox = None
        self.track_frame_count = 0

        # Statistics
        self.frame_count = 0
        self.detection_used = 0
        self.tracking_used = 0
        self.detection_rejected = 0
        self.both_failed = 0
        self.track_computed = 0
        self.track_cached = 0

        # Warmup YOLO
        print("Warming up YOLO...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detector.predict(dummy, verbose=False)

        print(f"Initialization complete (tracking interval: {tracking_interval}, jump threshold: {jump_threshold})")

    def init_tracker(self, frame, bbox):
        """Initialize DaSiamRPN tracker"""
        x, y, w, h = bbox

        # Clamp bbox to image boundary
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))

        if w < 3 or h < 3:
            return False

        try:
            # Convert bbox to center-size format
            cx = x + w / 2.0
            cy = y + h / 2.0
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])

            # Initialize DaSiamRPN state
            self.dasiamrpn_state = SiamRPN_init(frame, target_pos, target_sz, self.net)
            self.tracking_initialized = True
            self.last_track_bbox = bbox

            return True

        except Exception as e:
            if not hasattr(self, '_init_error_printed'):
                print(f"DaSiamRPN initialization failed: {e}")
                self._init_error_printed = True
            return False

    def track(self, frame):
        """Update DaSiamRPN tracker"""
        if not self.tracking_initialized or self.dasiamrpn_state is None:
            return None

        try:
            # Run tracker update
            self.dasiamrpn_state = SiamRPN_track(self.dasiamrpn_state, frame)
            self.track_computed += 1

            # Read tracking result
            target_pos = self.dasiamrpn_state['target_pos']
            target_sz = self.dasiamrpn_state['target_sz']

            # Convert to [x, y, w, h]
            cx, cy = target_pos[0], target_pos[1]
            w, h = target_sz[0], target_sz[1]

            x = cx - w / 2.0
            y = cy - h / 2.0

            tracked_bbox = [int(x), int(y), int(w), int(h)]

            # Simple validity check
            h_frame, w_frame = frame.shape[:2]
            if (x < -100 or y < -100 or w < 3 or h < 3 or
                x > w_frame + 100 or y > h_frame + 100):
                self.tracking_initialized = False
                return None

            self.last_track_bbox = tracked_bbox
            return tracked_bbox

        except Exception:
            self.tracking_initialized = False
            return None

    def detect(self, frame):
        """Run YOLO detection"""
        results = self.detector.predict(
            frame,
            conf=self.conf_threshold,
            verbose=False
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, 0.0

        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))

        box = boxes[best_idx]
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])

        x1, y1, x2, y2 = xyxy
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

        return bbox, conf

    def calculate_distance(self, bbox1, bbox2):
        """Calculate center distance between two bboxes"""
        if bbox1 is None or bbox2 is None:
            return float('inf')

        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        cx1 = x1 + w1 / 2.0
        cy1 = y1 + h1 / 2.0
        cx2 = x2 + w2 / 2.0
        cy2 = y2 + h2 / 2.0

        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def process_frame(self, frame):
        """
        Processing logic:
        - YOLO detection runs every frame
        - DaSiamRPN runs every N frames when detection is available
        - DaSiamRPN runs every frame when detection fails
        - Cached tracker output is used between sparse updates
        """
        self.frame_count += 1

        # Detection every frame
        det_bbox, det_conf = self.detect(frame)

        # Tracking behavior depends on detection availability
        track_bbox = None
        if self.tracking_initialized:
            if det_bbox is None:
                # Detection failed, so run tracker immediately
                track_bbox = self.track(frame)
                self.track_computed += 1
            else:
                # Detection succeeded, sparse tracking is enough
                self.track_frame_count += 1

                if self.track_frame_count >= self.tracking_interval:
                    # Periodic tracker validation
                    track_bbox = self.track(frame)
                    self.track_computed += 1
                    self.track_frame_count = 0
                else:
                    # Use cached tracking result between updates
                    track_bbox = self.last_track_bbox
                    self.track_cached += 1

        # Final decision
        final_bbox = None
        final_conf = 0.0
        status = 'LOST'

        if det_bbox is not None and track_bbox is not None:
            # Use tracker to validate detection
            distance = self.calculate_distance(det_bbox, track_bbox)

            if distance > self.jump_threshold:
                # Detection moved too much, keep tracker result
                final_bbox = track_bbox
                status = 'TRACKED (det rejected)'
                self.tracking_used += 1
                self.detection_rejected += 1
            else:
                # Detection is accepted
                final_bbox = det_bbox
                final_conf = det_conf
                status = 'DETECTED'
                self.detection_used += 1

                # Refresh tracker with accepted detection
                self.init_tracker(frame, det_bbox)
                self.track_frame_count = 0

        elif det_bbox is not None:
            # Detection only
            final_bbox = det_bbox
            final_conf = det_conf
            status = 'DETECTED'
            self.detection_used += 1

            # Initialize tracker from detection
            self.init_tracker(frame, det_bbox)
            self.track_frame_count = 0

        elif track_bbox is not None:
            # Tracking only
            final_bbox = track_bbox
            status = 'TRACKED (no det)'
            self.tracking_used += 1

        else:
            # Detection and tracking both failed
            status = 'LOST'
            self.both_failed += 1

        return final_bbox, status, final_conf


def draw_bbox(frame, bbox, status, conf, fps, frame_idx, stats):
    """Draw bbox and tracking information"""

    # Color by state
    colors = {
        'DETECTED': (0, 255, 0),
        'TRACKED (no det)': (255, 165, 0),
        'TRACKED (det rejected)': (255, 100, 0),
        'LOST': (0, 0, 255)
    }
    color = colors.get(status, (255, 255, 255))

    # Draw bbox
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        print("Center:", x + w / 2, y + h / 2)

        # Label
        if conf > 0:
            label = f"{status} {conf:.2f}"
        else:
            label = status

        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Show bbox size
        size_text = f"{w}x{h}"
        cv2.putText(frame, size_text, (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Frame index
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Running statistics
    if stats['frame_count'] > 0:
        det_pct = stats['detection_used'] / stats['frame_count'] * 100
        trk_pct = stats['tracking_used'] / stats['frame_count'] * 100
        rej_pct = stats['detection_rejected'] / stats['frame_count'] * 100
        fail_pct = stats['both_failed'] / stats['frame_count'] * 100

        y_pos = 90
        cv2.putText(frame, f"Det: {det_pct:.0f}%", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        y_pos += 25
        cv2.putText(frame, f"Track: {trk_pct:.0f}%", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        y_pos += 25
        cv2.putText(frame, f"Reject: {rej_pct:.0f}%", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        # Cached tracking usage
        if stats['track_computed'] + stats['track_cached'] > 0:
            y_pos += 25
            cached_pct = stats['track_cached'] / (stats['track_computed'] + stats['track_cached']) * 100
            cv2.putText(frame, f"Cached: {cached_pct:.0f}%", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 2)

    return frame


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=640,
    display_height=640,
    framerate=30,
    flip_method=0,
):
    """GStreamer pipeline for CSI camera"""
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def manual_init_select_bbox(cap):
    """
    Manual bbox selection at startup

    Controls:
    - n : next frame
    - s : select ROI
    - q / ESC : cancel and use auto detection
    """
    print("\n" + "=" * 60)
    print("Manual initialization mode")
    print("=" * 60)
    print("Controls:")
    print("  n : next frame")
    print("  s : select ROI")
    print("  q or ESC : cancel and continue with auto detection")
    print("=" * 60 + "\n")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            return None, None

        frame_idx += 1

        # Show current frame with instructions
        display = frame.copy()
        cv2.putText(display, f"Frame {frame_idx} | 'n':next  's':select  'q':cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, "Move to the desired frame and press 's' to select the target",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Manual Init - Select Target", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('n'):
            continue

        elif key == ord('s'):
            print("Draw a bounding box around the target.")
            bbox = cv2.selectROI("Manual Init - Select Target", frame,
                                 fromCenter=False, showCrosshair=True)
            x, y, w, h = [int(v) for v in bbox]

            if w > 0 and h > 0:
                print(f"Selected bbox: x={x}, y={y}, w={w}, h={h}")
                cv2.destroyWindow("Manual Init - Select Target")
                return frame, [x, y, w, h]
            else:
                print("Invalid bbox. Please try again.")
                continue

        elif key == ord('q') or key == 27:
            print("Manual initialization cancelled. Auto detection will be used.")
            cv2.destroyWindow("Manual Init - Select Target")
            return None, None

    return None, None


def run_inference(tracker, source, output_path=None, use_csi=False, no_display=False, manual_init=False):
    """Run inference"""

    # Open source
    if isinstance(source, int):
        camera_id = source

        if use_csi:
            print(f"Opening CSI camera {camera_id}")
            cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=camera_id), cv2.CAP_GSTREAMER)
        else:
            print(f"Opening USB camera {camera_id}")
            cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        if source.isdigit():
            camera_id = int(source)
            if use_csi:
                print(f"Opening CSI camera {camera_id}")
                cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=camera_id), cv2.CAP_GSTREAMER)
            else:
                print(f"Opening USB camera {camera_id}")
                cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
                cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            import os
            video_path = os.path.expanduser(source)
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                return

            print(f"Opening video file: {video_path}")
            cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open source")
        return

    # Input info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Input: {width}x{height} @ {fps}fps")

    # Optional manual initialization
    if manual_init:
        init_frame, init_bbox = manual_init_select_bbox(cap)
        if init_frame is not None and init_bbox is not None:
            print("Initializing tracker with manual bbox")
            tracker.init_tracker(init_frame, init_bbox)
            print("Tracker initialized. Starting tracking.")
        else:
            print("Manual initialization skipped. Auto detection will be used.")

    # Output writer
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")

    # FPS buffer
    fps_history = deque(maxlen=30)

    print("Starting processing (ESC to quit)")
    print("\n" + "=" * 60)
    print("Tracking strategy:")
    print(" - YOLO detection runs every frame")
    if tracker.tracking_initialized:
        print(" - Tracker initialized manually")
    print(f" - DaSiamRPN runs every {tracker.tracking_interval} frames")
    print(" - Cached tracking result used between updates")
    print("=" * 60 + "\n")

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Main processing
            bbox, status, conf = tracker.process_frame(frame)

            # FPS
            elapsed = time.time() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(current_fps)
            avg_fps = np.mean(fps_history)

            # Stats
            stats = {
                'frame_count': tracker.frame_count,
                'detection_used': tracker.detection_used,
                'tracking_used': tracker.tracking_used,
                'detection_rejected': tracker.detection_rejected,
                'both_failed': tracker.both_failed,
                'track_computed': tracker.track_computed,
                'track_cached': tracker.track_cached
            }

            # Visualization
            frame = draw_bbox(frame, bbox, status, conf, avg_fps, tracker.frame_count, stats)

            # Display
            if not no_display:
                cv2.imshow("Sparse DaSiamRPN Tracker", frame)

            # Save frame
            if out:
                out.write(frame)

            # Exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\nProcess interrupted")

    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        # Summary
        print("\n" + "=" * 60)
        print("Performance summary")
        print("=" * 60)
        print(f"Total frames: {tracker.frame_count}")

        if tracker.frame_count > 0:
            det_pct = tracker.detection_used / tracker.frame_count * 100
            trk_pct = tracker.tracking_used / tracker.frame_count * 100
            rej_pct = tracker.detection_rejected / tracker.frame_count * 100
            fail_pct = tracker.both_failed / tracker.frame_count * 100

            print(f"\nDetection: {tracker.detection_used} ({det_pct:.1f}%)")
            print(f"Tracking: {tracker.tracking_used} ({trk_pct:.1f}%)")
            print(f"Rejected: {tracker.detection_rejected} ({rej_pct:.1f}%)")
            print(f"Lost: {tracker.both_failed} ({fail_pct:.1f}%)")

            print(f"\nDaSiamRPN computed: {tracker.track_computed}")
            print(f"DaSiamRPN cached: {tracker.track_cached}")
            if tracker.track_computed + tracker.track_cached > 0:
                cached_pct = tracker.track_cached / (tracker.track_computed + tracker.track_cached) * 100
                print(f"Cache efficiency: {cached_pct:.1f}%")

        if fps_history:
            print(f"\nAverage FPS: {np.mean(fps_history):.1f}")
            print(f"Min FPS: {np.min(fps_history):.1f}")
            print(f"Max FPS: {np.max(fps_history):.1f}")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Optimized Hybrid Detection + DaSiamRPN with Manual Init'
    )

    parser.add_argument('--model', type=str, default='best_fp16.engine',
                        help='YOLO model path')
    parser.add_argument('--dasiamrpn-model', type=str,
                        default='./DaSiamRPN/code/SiamRPNVOT.model',
                        help='DaSiamRPN model path')
    parser.add_argument('--source', type=str, default='0',
                        help='Video file or camera ID')
    parser.add_argument('--output', type=str,
                        help='Output video path')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--jump-threshold', type=int, default=200,
                        help='Max allowed detection jump (pixels)')
    parser.add_argument('--tracking-interval', type=int, default=3,
                        help='Run DaSiamRPN every N frames (default: 3)')
    parser.add_argument('--manual-init', action='store_true',
                        help='Enable manual bbox selection at start')
    parser.add_argument('--csi', action='store_true',
                        help='Use CSI camera')
    parser.add_argument('--no-display', action='store_true',
                        help='Headless mode')

    args = parser.parse_args()

    # Initialize tracker
    tracker = SparseDaSiamRPNTracker(
        detector_path=args.model,
        conf_threshold=args.conf,
        jump_threshold=args.jump_threshold,
        dasiamrpn_model=args.dasiamrpn_model,
        tracking_interval=args.tracking_interval
    )

    # Parse source
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    # Run inference
    run_inference(
        tracker,
        source=source,
        output_path=args.output,
        use_csi=args.csi,
        no_display=args.no_display,
        manual_init=args.manual_init
    )


if __name__ == "__main__":
    main()
