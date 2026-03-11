export PATH="/home/runbk0401/miniconda3/envs/openmmlab/bin:$PATH"


python test_drone_detector.py     --model anti_drone_v2/yolo11n_v2/weights/best.pt     --webcam


python test_drone_detector.py     --model anti_drone_v2/yolo11n_v2/weights/best.pt     --video yolo_test.mp4


python test_drone_detector.py     --model anti_drone_v2/yolo11n_v2/weights/best.pt     --video "https://www.youtube.com/watch?app=desktop&v=6AYQUrdrYjc"


python test_drone_detector.py     --model anti_drone_v2/yolo11n_v2/weights/best.pt     --video test6_640.mp4 --tracker siamrpn

python test_drone_detector.py     --model anti_drone_v2/yolo11n_v2/weights/best.pt     --video test11_640.mp4



python test_drone_detector_analysis.py     --model anti_drone_v2/yolo11n_v2/weights/best.pt   
  --video test11_640.mp4 --analysis basline.json


python test_drone_detector_MK2.py \
  --model anti_drone_v2/yolo11n_v2/weights/best.pt \
  --video test6_640.mp4 \
  --tracker siamrpn \
  --siamrpn_model /path/to/dasiamrpn_model.onnx \
  --siamrpn_kernel /path/to/dasiamrpn_kernel_r1.onnx



yolo detect train \
    data=~/Downloads/NEST_YOLO_v2/data.yaml \
    model=yolo11s.pt \
    epochs=150 \
    batch=8 \
    imgsz=640 \
    device=0 \
    patience=50 \
    project=anti_drone_v2 \
    name=yolo11s \
    save=True \
    plots=True


yolo detect train \
    data=~/Downloads/NEST_YOLO_v2/data.yaml \
    model=yolo11n.pt \
    epochs=150 \
    batch=8 \
    imgsz=640 \
    device=0 \
    patience=50 \
    project=anti_drone_v2 \
    name=yolo11n_v3 \
    save=True \
    plots=True


yolo detect train \
  data=~/Downloads/NEST_YOLO_v2/data.yaml \
  model=yolo11n.pt \
  epochs=200 \
  batch=8 \
  imgsz=640 \
  device=0 \
  patience=50 \
  project=anti_drone_v2 \
  name=yolo11n_640_smallobj \
  close_mosaic=10 \
  mosaic=0.8 \
  mixup=0.10 \
  copy_paste=0.10 \
  hsv_h=0.02 hsv_s=0.55 hsv_v=0.45 \
  degrees=5 translate=0.15 scale=0.60 perspective=0.0005 \
  fliplr=0.5 flipud=0.2


python test_drone_detector_MK42.py     --model anti_drone_v2/yolo11s/weights/best.pt     --video test11_640.mp4 --tracker dasiamrpn --dasiam_model SiamRPNVOT.model 


python test_drone_detector_MK43.py     --model anti_drone_v2/yolo11s/weights/best.pt --manual_init     --video test11_640.mp4 --tracker dasiamrpn --dasiam_model SiamRPNVOT.model

python test_drone_detector_MK43.py     --model anti_drone_v2/yolo11s/weights/best.pt --manual_init     --video test11_640.mp4 --dasiam_model SiamRPNVOT.model --tracker opencv

python test_drone_detector_MK44.py     --model anti_drone_v2/yolo11s/weights/best.pt --manual_init     --video test11_640.mp4 --dasiam_model SiamRPNVOT.model --tracker opencv


# This seems like it it best? It has padding and uses CSRT
python test_drone_detector_MK44.py     --model anti_drone_v2/yolo11s/weights/best.pt --manual_init     --video test11_640.mp4  --tracker opencv

python test_drone_detector_MK44.py     --model anti_drone_v2/yolo11s/weights/best.pt --manual_init --tracker dasiamrpn --dasiam_model SiamRPNVOT.model --video test6.mp4


python test_drone_detector_MK44.py     --model anti_drone_v2/yolo11s/weights/best.pt --manual_init --tracker opencv --video test1.mp4 

 python kalman_drone_tracker.py     --model anti_drone_v2/yolo11s/weights/best.pt   --video test11_640.mp4  --interval 5