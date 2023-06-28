from ultralytics import YOLO
import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
if __name__ == '__main__':
    model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
   model.train(data="./data.yaml", epochs=300, imgsz=640, device='0', workers=4)
   # model = YOLO("./runs/detect/train6/weights/best.pt")
   # model.val(data="data.yaml", imgsz=640, batch=16, conf=0.001, iou=0.6, save_json=True, model=model)
