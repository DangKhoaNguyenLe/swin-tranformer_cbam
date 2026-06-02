import cv2
from ultralytics import YOLO
import sys
import glob

model = YOLO('yolov8n-face.pt')
imgs = glob.glob('dataset/*/*_orig.jpg')
if not imgs:
    print("No images found")
    sys.exit()

img = cv2.imread(imgs[0])
res = model(img)
for r in res:
    if hasattr(r, 'keypoints') and r.keypoints is not None:
        print("Keypoints:", r.keypoints.xy.shape)
    else:
        print("No keypoints")
