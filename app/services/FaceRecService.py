# -*- coding: utf-8 -*-

# 视频眨眼检测
# 导入工具包
from collections import OrderedDict
from keras.models import load_model
from scipy.spatial import distance as dist
import numpy as np
import cv2
import os
import paddlehub as hub
# 设置判断参数

from models.src.anti_spoof_predict import AntiSpoofPredict


# 关键点排序
from models.src.generate_patches import CropImage
from models.src.utility import parse_model_name

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("left_eyebrow", (17, 22)),
    ("right_eyebrow", (22, 27)),
    ("left_eye", (36, 42)),
    ("right_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])
EYE_AR_THRESH = 0.3 # 低于该值则判断为眨眼
BLINK_THRESH = 1  # 眨眼次数的阈值
EYE_AR_CONSEC_FRAMES = 3
SCALE_WIDTH = 320

mask_detector = hub.Module(name="pyramidbox_lite_mobile_mask")
face_landmark = hub.Module(name="face_landmark_localization")
caffemodel = "../../models/detection_model/Widerface-RetinaFace.caffemodel"
deploy = "../../models/detection_model/deploy.prototxt"
as_model = AntiSpoofPredict(0,caffemodel,deploy)
model = load_model("../../models/fas.h5")
def eye_aspect_ratio(eye):
    """
    计算眼睛上下关键点欧式距离
    :param eye:眼睛关键点位置
    :return: 眼睛睁开程度
    """
    # 计算距离，竖直的
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算距离，水平的
    C = dist.euclidean(eye[0], eye[3])
    # ear值
    ear = (A + B) / (2.0 * C)
    return ear

def as_check(image, model_dir='../../models/anti_spoof_models'):
    image_cropper = CropImage()
    image_bbox = as_model.get_bbox(image)
    prediction = np.zeros((1, 3))
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += as_model.predict(img, os.path.join(model_dir, model_name))

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    return (label,value,image_bbox)

def fas_check(X):
    X = (cv2.resize(X,(224,224))-127.5)/127.5
    t = model.predict(np.array([X]))[0]
    return t

def check_video_uri(video_uri,userinfo = ''):
    # 读取视频
    TOTAL_BLINK = 0
    COUNTER = 0
    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
    print("[INFO] starting video stream thread...")
    # 湖南直播数据rtmp://58.200.131.2:1935/livetv/hunantv
    # print(video_uri)
    vs = cv2.VideoCapture(0)
    rate = vs.get(cv2.CAP_PROP_FPS)
    print(rate)
    # 遍历每一帧
    flag = False
    while True:
        # 预处理
        frame = vs.read()[1]
        if frame is None:
            break
        (h, w) = frame.shape[:2]
        if h > w :
            width = 300
        else:
            width = 600
        r = width / float(w)
        dim = (width, int(h * r))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # 检测人脸
        rects = mask_detector.face_detection([frame])

        if len(rects[0]['data']) != 1:
            COUNTER = 0
        else:
            # 遍历每一个检测到的人脸
            for data in rects[0]['data']:
                h = data['bottom'] - data['top']
                w = data['right'] - data['left']
                _r = int(max(w,h)*0.6)
                cx,cy = (data['left']+data['right'])//2, (data['top']+data['bottom'])//2

                x1 = cx - _r
                y1 = cy - _r

                x1 = int(max(x1,0))
                y1 = int(max(y1,0))

                x2 = cx + _r
                y2 = cy + _r

                h,w,c =frame.shape
                x2 = int(min(x2 ,w-2))
                y2 = int(min(y2, h-2))

                _frame = frame[y1:y2 , x1:x2]
                value = fas_check(_frame)
                if value > 0.95:
                    result_text = "RealFace Score: {:.2f}".format(float(value))
                    color = (255, 0, 0)
                else:
                    result_text = "FakeFace Score: {:.2f}".format(float(value))
                    color = (0, 0, 255)
                cv2.rectangle(
                    frame,
                    (x1,y1) ,(x2,y2),
                    color, 2)
                cv2.putText(
                    frame,
                    result_text,
                    (x1,y1-5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5*frame.shape[0]/512, color)
                cv2.rectangle(frame, (x1,y1) ,(x2,y2) , (0,255,0)  ,2)

                if fas_check(_frame) > 0.95:
                    shape = face_landmark.keypoint_detection([frame])
                    if len(shape) == 0:
                        continue
                    landmark = shape[0]['data'][0]
                    # print(landmark)
                    # 分别计算ear值
                    leftEye = landmark[lStart:lEnd]
                    rightEye = landmark[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    # 算一个平均的
                    ear = (leftEAR + rightEAR) / 2.0
                    # print(ear)
                    # 检查是否满足阈值
                    # for i in landmark:
                    #     cv2.circle(frame, (int(i[0]),int(i[1])), 1, (0, 0, 255), 2)
                    if ear < EYE_AR_THRESH:
                        if flag:
                            COUNTER += 1
                            # 如果连续几帧都是闭眼的，总数算一次
                            if COUNTER > EYE_AR_CONSEC_FRAMES:
                                flag = False
                                TOTAL_BLINK += 1
                                # 重置
                                COUNTER = 0
                                if TOTAL_BLINK > BLINK_THRESH:
                                    vs.release()
                                    return TOTAL_BLINK

                    else:
                        flag = True
                        COUNTER = 0
                    # cv2.putText(frame, "Blinks: {}".format(TOTAL_BLINK), (10, 30),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # cv2.putText(frame, "EAR: {:.2f}".format(ear), (150, 30),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow(video_uri,frame)
        cv2.waitKey(int(rate))
    vs.release()
    return TOTAL_BLINK

# def check_imgfiles(imgfiles, userinfo = ''):
#     # 读取视频
#     TOTAL_BLINK = 0
#     COUNTER = 0
#     (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
#     (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
#     print("[INFO] starting load image frames...")
#     flag = False
#     # 遍历每一帧
#     for file in imgfiles:
#         file_bytes = file.read()
#         frame = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype='uint8'), cv2.IMREAD_COLOR)
#         (h, w) = frame.shape[:2]
#         if h > w :
#             width = 300
#         else:
#             width = 600
#         r = width / float(w)
#         dim = (width, int(h * r))
#         frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # 检测人脸
#         rects = mask_detector.face_detection(images=[frame])
#         if len(rects[0]['data']) != 1:
#             COUNTER = 0
#         else:
#             # 遍历每一个检测到的人脸
#             for data in rects[0]['data']:
#                 h = (data['bottom'] - data['top'])
#                 w = (data['right'] - data['left'])
#                 _r = int(max(w,h)*0.6)
#                 cx,cy = (data['left']+data['right'])//2, (data['top']+data['bottom'])//2
#
#                 x1 = cx - _r
#                 y1 = cy - _r
#
#                 x1 = int(max(x1,0))
#                 y1 = int(max(y1,0))
#
#                 x2 = cx + _r
#                 y2 = cy + _r
#
#                 h,w,c =frame.shape
#                 x2 = int(min(x2 ,w-2))
#                 y2 = int(min(y2, h-2))
#
#                 _frame = frame[y1:y2 , x1:x2]
#
#                 # print(self.fas_check(_frame))
#                 if fas_check(_frame) > 0.95:
#                     # 获取坐标
#                     shape = face_landmark.keypoint_detection([frame])
#                     landmark = shape[0]['data'][0]
#                     # print(landmark)
#                     # 分别计算ear值
#                     leftEye = landmark[lStart:lEnd]
#                     rightEye = landmark[rStart:rEnd]
#                     leftEAR = eye_aspect_ratio(leftEye)
#                     rightEAR = eye_aspect_ratio(rightEye)
#                     # 算一个平均的
#                     ear = (leftEAR + rightEAR) / 2.0
#                     # print(ear)
#                     # 检查是否满足阈值
#                     if ear < EYE_AR_THRESH:
#                         if flag:
#                             COUNTER += 1
#                             # 如果连续几帧都是闭眼的，总数算一次
#                             if COUNTER > EYE_AR_CONSEC_FRAMES:
#                                 flag = False
#                                 TOTAL_BLINK += 1
#                                 # 重置
#                                 COUNTER = 0
#                                 if TOTAL_BLINK > BLINK_THRESH:
#                                     return TOTAL_BLINK
#                     else:
#                         flag = True
#                         COUNTER = 0
#
#     return TOTAL_BLINK

def LivenessCheck(img_files, video_uri, userinfo):
    if len(img_files) > 0:
        print(len(img_files))
        # result = check_imgfiles(img_files, userinfo)
    elif video_uri is not None:
        result = check_video_uri(video_uri, userinfo)
    else:
        return 'no'

    if result > BLINK_THRESH:
        return 'yes'
    else:
        return 'no'

if __name__ == '__main__':
    LivenessCheck([],'test','')
