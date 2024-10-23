import cv2
import numpy as np
from onnxruntime import InferenceSession

# ONNX 모델 경로
model_path = 'weights/mpipe_bface_boxes_ops16.onnx'
sess = InferenceSession(model_path)

# 입력 및 출력 정보 출력 (테스트용)
for i in sess.get_inputs():
    print("Expected input -", i.name, "- shape: ", i.shape)
for o in sess.get_outputs():
    print("Expected output -", o.name, "- shape: ", o.shape)


#
def resize_image(image, re_size, keep_ratio=True):
    """
    Resize image while keeping aspect ratio and adding black padding by creating
    a black background using np.zeros and overlaying the resized image on it.
    Args:
        image: origin image (numpy array)
        re_size: resize target size (width, height)
        keep_ratio: keep aspect ratio. Default is set to true.
    Returns:
        re_image: resized image with black padding (if necessary)
        resize_ratio: resize ratio used
    """
    h, w = image.shape[:2]

    if not keep_ratio:
        # 비율을 유지하지 않고 바로 리사이즈
        re_image = cv2.resize(image, (re_size[0], re_size[1])).astype('float32')
        return re_image, 1.0

    # 리사이즈 비율 계산
    target_ratio = re_size[0] / re_size[1]  # 목표 비율 (가로/세로)
    image_ratio = w / h  # 원본 이미지 비율 (가로/세로)

    if image_ratio > target_ratio:
        # 이미지가 더 넓은 경우 (너비 기준으로 리사이즈)
        resize_ratio = re_size[0] / w
        re_w, re_h = re_size[0], int(h * resize_ratio)
    else:
        # 이미지가 더 높은 경우 (높이 기준으로 리사이즈)
        resize_ratio = re_size[1] / h
        re_w, re_h = int(w * resize_ratio), re_size[1]

    # 비율을 유지한 리사이즈
    re_image = cv2.resize(image, (re_w, re_h)).astype('float32')

    # 검은색 배경 이미지를 먼저 생성 (zeros로 채워진 이미지)
    padded_image = np.zeros((re_size[1], re_size[0], 3), dtype=np.float32)

    # 리사이즈된 이미지를 검은색 배경 위에 덮어씌움
    # 이미지를 왼쪽 상단에 배치하고, 남은 영역은 검은색으로 유지
    padded_image[0:re_h, 0:re_w, :] = re_image

    return padded_image, resize_ratio


# 이미지에서 감지된 얼굴을 그리는 함수
def draw_detections(img, detections, with_keypoints=True):
    if isinstance(detections, np.ndarray):
        if detections.ndim == 1:
            detections = np.expand_dims(detections, axis=0)

        maximage = img.shape[0]

        for i in range(detections.shape[0]):
            ymin = int(detections[i, 0] * maximage)
            xmin = int(detections[i, 1] * maximage)
            ymax = int(detections[i, 2] * maximage)
            xmax = int(detections[i, 3] * maximage)

            # 얼굴 박스 그리기
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # 키포인트 그리기 (눈, 코, 입 등)
            if with_keypoints:
                for k in range(6):
                    kp_x = int(detections[i, 4 + k * 2] * maximage)
                    kp_y = int(detections[i, 4 + k * 2 + 1] * maximage)
                    cv2.circle(img, (kp_x, kp_y), 2, (255, 0, 0), -1)


# 웹캠에서 실시간으로 이미지 캡처
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    y_c, x_c = frame.shape[:2]

    # # Y축이 길어지면 바운딩박스가 세로로 길어짐
    x_c = x_c//2
    y_c = y_c//2
    frame = frame[y_c-400:y_c+400, x_c-300:x_c+300, :]

    # # X축이 길어지면 바운딩박스가 가로로 길어짐
    # x_c = x_c//2
    # y_c = y_c//2
    # frame = frame[y_c-300:y_c+300, x_c-500:x_c+500, :]

    # 이미지를 RGB로 변환
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 모델에 입력하기 위한 전처리
    img = img / 127.5 - 1.0

    # pimg = cv2.resize(img, (128, 128))
    pimg, resize_ratio = resize_image(img, (128, 128), keep_ratio=True)
    # pimg = pimg / 127.5 - 1.0

    pimg = np.expand_dims(pimg, axis=0).astype(np.float32)

    # 모델 추론
    out3 = sess.run(None, {'input': pimg, 'conf_threshold': [.3], 'max_detections': [100], 'iou_threshold': [.5]})

    # 감지된 얼굴 그리기
    draw_detections(frame, out3[0][0], resize_ratio)

    # 결과를 cv2를 통해 실시간으로 출력
    cv2.imshow('Face Detection', frame)

    # ESC 키를 눌러 종료
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
