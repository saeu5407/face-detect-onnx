'''https://huggingface.co/amd/retinaface/tree/main'''

import os
import cv2
import onnxruntime as ort
import torch
import numpy as np
from itertools import product as product
from math import ceil

# 모델 설정
CFG = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
}
INPUT_SIZE = [608, 640]
DEVICE = torch.device("cpu")


# PriorBox 클래스 정의
class PriorBox(object):
    def __init__(self, cfg, image_size):
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                    cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                    anchors.append([cx, cy, s_kx, s_ky])
        return torch.Tensor(anchors).view(-1, 4)


# NMS 함수
def py_cpu_nms(dets, thresh):
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


# 디코딩 함수
def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


# 전처리 함수
def preprocess(img_raw, input_size, device):
    img = np.float32(img_raw)
    img, resize = resize_image(img, input_size)
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    return img.numpy(), scale.to(device), resize


# 이미지 크기 조정 및 패딩 함수
# 비율에 따라서 다르게 작업하는데 일단은 그냥 돌아간다고 치면 이 함수는 그냥 리사이즈만 써도 될 듯
def resize_image(image, re_size):
    h, w = image.shape[0:2]
    ratio = re_size[0] / re_size[1]
    if h / w <= ratio:
        resize_ratio = re_size[1] / w
        re_h, re_w = int(h * resize_ratio), re_size[1]
    else:
        resize_ratio = re_size[0] / h
        re_h, re_w = re_size[0], int(w * resize_ratio)

    resized_image = cv2.resize(image, (re_w, re_h)).astype('float32')
    pad_image = cv2.copyMakeBorder(resized_image, 0, re_size[0] - re_h, 0, re_size[1] - re_w, cv2.BORDER_CONSTANT,
                                   value=(0.0, 0.0, 0.0))
    return pad_image, resize_ratio


# 후처리 함수
def postprocess(cfg, img, outputs, scale, resize, confidence_threshold, nms_threshold, device):
    loc, conf = torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])
    conf = torch.softmax(conf, dim=-1)
    priorbox = PriorBox(cfg, image_size=(img.shape[2], img.shape[1]))
    priors = priorbox.forward().to(device)
    boxes = decode(loc.squeeze(0), priors.data, cfg["variance"]) * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # 임계값 이상만 선택
    inds = np.where(scores > confidence_threshold)[0]
    boxes, scores = boxes[inds], scores[inds]

    # NMS 수행
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    return dets[keep, :]


# 추론 및 바운딩 박스, 정확도 출력 함수
def Retinaface_inference(run_ort, args):
    img_raw = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    print(img_raw)
    img, scale, resize = preprocess(img_raw, INPUT_SIZE, DEVICE)
    img = np.transpose(img, (0, 2, 3, 1))  # NHWC로 변환
    outputs = run_ort.run(None, {run_ort.get_inputs()[0].name: img})
    dets = postprocess(CFG, img, outputs, scale, resize, args.confidence_threshold, args.nms_threshold, DEVICE)

    # 결과 출력
    for i in range(len(dets)):
        box = dets[i, :4]
        confidence = dets[i, 4]
        print(f"Bounding Box: {box}, Confidence: {confidence}")

    return dets[:, :4], dets[:, 4:5]  # 바운딩 박스, 정확도 반환


# 실행 코드
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Retinaface")
    parser.add_argument("--trained_model", default="weights/RetinaFace_int.onnx", type=str)
    parser.add_argument("--image_path", default="data/test_image.png", type=str)
    parser.add_argument("--confidence_threshold", default=0.5, type=float)
    parser.add_argument("--nms_threshold", default=0.5, type=float)

    args = parser.parse_args()

    print(f"Loading model from {args.trained_model}")
    run_ort = ort.InferenceSession(args.trained_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # 추론 및 결과 출력
    boxes, confidences = Retinaface_inference(run_ort, args)
    print("Inference complete!")