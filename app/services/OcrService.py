import paddlehub as hub
import cv2
import numpy as np

ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")

def ImgOcr(img_files, img_uri):
    results = []
    if len(img_files) > 0:
        print(len(img_files))
        for file in img_files:
            file_bytes = file.read()
            image = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype='uint8'), cv2.IMREAD_COLOR)
            data = ocrByPreModel(image)
            results.append((file.filename,data))
    elif img_uri is not None:
        video_capture = cv2.VideoCapture(img_uri)
        while True:
            flag, frame = video_capture.read()
            if flag:
                data = ocrByPreModel(frame)
                results.append((img_uri,data))
            else:
                break
    else:
        return ''

    return results

def ocrByPreModel(np_image):
    np_images = []
    np_images.append(np_image)
    results = ocr.recognize_text(
                        images=np_images,         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                        use_gpu=False,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
                        visualization=False,       # 是否将识别结果保存为图片文件；
                        box_thresh=0.7,           # 检测文本框置信度的阈值；
                        text_thresh=0.9)          # 识别中文文本置信度的阈值；


    for result in results:
        return result['data']

