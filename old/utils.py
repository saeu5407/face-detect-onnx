import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import onnxruntime as ort
import torch
import cv2
from itertools import product
from math import ceil

# 설정 값 정의 (CFG에서 가져온 값)
MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
STEPS = [8, 16, 32]
VARIANCE = [0.1, 0.2]


# PriorBox와 관련된 anchor box 생성 함수
def generate_prior_boxes(min_sizes, steps, image_size):
    anchors = []
    for k, f in enumerate([[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]):
        min_size = min_sizes[k]
        step = steps[k]
        for i, j in product(range(f[0]), range(f[1])):
            for size in min_size:
                s_kx = size / image_size[1]
                s_ky = size / image_size[0]
                cx = (j + 0.5) * step / image_size[1]
                cy = (i + 0.5) * step / image_size[0]
                anchors.append([cx, cy, s_kx, s_ky])
    return np.array(anchors, dtype=np.float32)


# PriorBox 텐서 생성
image_size = [608, 640]
anchors = generate_prior_boxes(MIN_SIZES, STEPS, image_size)

# ONNX 텐서 생성 (anchors 값을 올바르게 넣어줌)
priorbox_tensor = helper.make_tensor(
    name='priors',
    data_type=TensorProto.FLOAT,
    dims=anchors.shape,
    vals=anchors.flatten().tolist()  # 값이 텐서 크기와 일치하도록 flatten()
)

# 모델 로드 및 PriorBox 연산 추가
onnx_model_path = "../weights/RetinaFace_int.onnx"
onnx_model = onnx.load(onnx_model_path)

# 입력 노드 정의 (loc와 scores 추가)
input_loc = helper.make_tensor_value_info('loc', TensorProto.FLOAT, [1, 16800, 4])  # loc 입력 정의
input_scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 16800, 2])  # scores 입력 정의

# PriorBox 노드를 Constant로 추가
priorbox_node = helper.make_node(
    'Constant',  # ONNX 상수 노드
    inputs=[],  # 입력 없음
    outputs=['priors'],  # 출력
    value=priorbox_tensor  # 미리 생성한 PriorBox 텐서를 사용
)

# priors를 loc와 같은 차원으로 맞추기 위해 Unsqueeze 연산 추가
unsqueeze_node = helper.make_node(
    'Unsqueeze',
    inputs=['priors'],
    outputs=['priors_unsqueezed'],
    axes=[0]  # 배치 차원 추가
)

# decode 및 NMS 노드를 추가하는 방식은 이전과 동일
decode_node = helper.make_node(
    'Add',  # 바운딩 박스 디코딩을 위한 연산
    inputs=['loc', 'priors_unsqueezed'],
    outputs=['decoded_boxes']
)

nms_node = helper.make_node(
    'NonMaxSuppression',  # NMS 연산
    inputs=['decoded_boxes', 'scores'],
    outputs=['selected_indices'],
    max_output_boxes_per_class=200,
    iou_threshold=0.5,
    score_threshold=0.4
)

# 그래프의 입력으로 loc와 scores 추가
onnx_model.graph.input.extend([input_loc, input_scores])

# PriorBox와 Unsqueeze, decode, NMS 노드를 그래프에 추가
onnx_model.graph.node.extend([priorbox_node, unsqueeze_node, decode_node, nms_node])

# 수정된 모델 저장
onnx.save(onnx_model, "RetinaFace_with_postprocessing.onnx")


# ONNX Runtime을 이용한 추론 함수 (동일)
def run_inference_with_onnx(image_path, model_path):
    ort_session = ort.InferenceSession(model_path)

    # 이미지 전처리
    image = cv2.imread(image_path)
    input_image = preprocess_image(image)

    # 추론 수행
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_image})

    # NMS 후처리 포함된 결과 반환
    boxes, scores = outputs[0], outputs[1]
    return boxes, scores


# 이미지 전처리 함수 (동일)
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 608))
    image = image.astype(np.float32)
    image -= (104, 117, 123)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image


# 데모 실행
if __name__ == "__main__":
    img_path = "test_image.jpg"
    model_path = "RetinaFace_with_postprocessing.onnx"

    # 모델 추론 및 결과 출력
    boxes, scores = run_inference_with_onnx(img_path, model_path)
    print(f"Bounding Boxes: {boxes}")
    print(f"Scores: {scores}")
