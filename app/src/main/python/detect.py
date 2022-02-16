from tensorflow.keras.models import load_model
from utils import get_yolo_boxes
from bbox  import draw_boxes
import numpy as np
import cv2
import io
import base64
from PIL import Image
from os.path import dirname, join
import tensorflow as tf

def det(data):
    decoded_data = base64.b64decode(data)
    np_data = np.fromstring(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Set network size along with obj_thresh and nms_thresh
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.65, 0.50
    labels = ["bicycle", "bus", "car", "motorbike", "person"]
    anchors = [20,37, 46,84, 57,185, 96,259, 126,120, 151,298, 229,349, 257,193, 362,352]

    # Load model
    filename = join(dirname(__file__), "voc_best214.h5")
    #infer_model = load_model(config['train']['saved_weights_name'])
    infer_model = load_model(filename)
    boxes = get_yolo_boxes(infer_model, [img_rgb], net_h, net_w, anchors, obj_thresh, nms_thresh)[0]
    draw_boxes(img_rgb, boxes, labels, obj_thresh)

    pil_im = Image.fromarray(img_rgb)
    buff = io.BytesIO()
    pil_im.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    return ""+str(img_str, 'utf-8')
