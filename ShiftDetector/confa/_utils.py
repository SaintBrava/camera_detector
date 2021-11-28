# import base64
import datetime as dt
# import re
# import struct
import sys
# import time
import traceback
# from contextlib import redirect_stdout
# from functools import wraps
# from json import JSONEncoder
# from typing import List

# import cv2
# import numpy as np
# import psutil as ps
# from PIL import Image

# import config


# def timer(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start = time.perf_counter()
#         result = func(*args, **kwargs)
#         stop = time.perf_counter()
#         print(f"{func.__name__!r} runtime: {(stop - start):.4f} s")
#         return result

#     return wrapper


# def logger(file):
#     def decorator(function):
#         @wraps(function)
#         def wrapper(*args, **kwargs):
#             with open(file, encoding="utf-8", mode="a") as fp:
#                 with redirect_stdout(fp):
#                     result = function(*args, **kwargs)
#                     return result

#         return wrapper

#     return decorator


# class NumpyEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, dt.datetime):
#             return str(obj)
#         return JSONEncoder.default(self, obj)


# def decode_image(img_buf: bytes) -> np.ndarray:
#     img_arr = np.frombuffer(img_buf, dtype=np.uint8)
#     img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
#     return img


# def encode_to_redis(array: np.ndarray, timestamp):
#     h, w, c = array.shape
#     shape = struct.pack('>IIId', h, w, c, timestamp)
#     encoded = shape + array.tobytes()
#     return encoded


# def decode_from_redis(byte_array):
#     h, w, c, timestamp = struct.unpack('>IIId', byte_array[:20])
#     array = np.frombuffer(byte_array[20:], dtype=np.uint8).reshape(h, w, c)
#     return array, timestamp


def timestamp_for_filename(datetime=None) -> str:
    if datetime is None:
        datetime = dt.datetime.now()
    if sys.platform != 'win32':
        cur_time = datetime.strftime("%Y-%m-%d_%H:%M:%S.%f")
    else:
        cur_time = datetime.strftime("%Y-%m-%d_%H-%M-%S.%f")
    return cur_time


# def calc_img_size(file: str, max_w: int = 750):
#     img = Image.open(file)
#     w, h = img.size
#     max_h = int(max_w / w * h)

#     return max_h, max_w


# def get_encoded_frame(frame):
#     img_h, img_w = calc_img_size(frame)
#     with open(str(frame), "rb") as image_file:
#         real_image = image_file.read()
#         img_b64 = base64.b64encode(real_image).decode("utf-8")
#         return img_h, img_w, f"data:image/jpg;base64,{img_b64}"


# def iou(box1, box2):
#     # print(box1, box2)
#     # determine the (x, y)-coordinates of the intersection rectangle
#     x11, y11, x12, y12 = box1
#     x21, y21, x22, y22 = box2
#     xA = max(x11, x21)
#     yA = max(y11, y21)
#     xB = min(x12, x22)
#     yB = min(y12, y22)

#     # compute the area of intersection rectangle
#     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
#     boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = interArea / float(boxAArea + boxBArea - interArea)

#     # return the intersection over union value
#     # print(iou)
#     return iou


# def save_img(encoded_img, save_path):
#     try:
#         encoded_img.tofile(str(save_path))
#         print("saved to", save_path)
#         return True
#     except:
#         print("can't save to", save_path)
#         return False


# def encode_img(img: np.ndarray, processing=None):
#     try:
#         save_img = img.copy()
#         # apply processing to img if have some
#         if processing is not None:
#             save_img = processing(save_img)
#         # save_img is BGR
#         is_success, buffer = cv2.imencode(".jpg", save_img)
#         return is_success, buffer
#     except:
#         traceback.print_exc()
#         return False, None


# def path_to_url(path):
#     try:
#         if config.URL_PREFIX:
#             save_path_str = str(path)
#             img_url = save_path_str.replace(str(config.VIDEO_FRAMES_PATH), config.URL_PREFIX)
#             return img_url
#     except:
#         print("can't generate url to frame", path)
#         return None


# class Fps:
#     def __init__(self, initial_value):
#         self.value = initial_value
#         self.acc_ts = []
#         self.acc_ts_diff = []
#         self.acc_num_frames = 0
#         self.max_samples = config.BATCH_SIZE * 10

#     def update(self, ts: List[float], num_frames: int):
#         if len(self.acc_ts) >= 1:
#             self.acc_ts_diff.append(ts[0] - self.acc_ts[-1])
#         for t1, t2 in zip(ts, ts[1:]):
#             self.acc_ts_diff.append(t2 - t1)
#         self.acc_ts.extend(ts)
#         self.acc_num_frames += num_frames

#         num_delete = len(self.acc_ts) - self.max_samples
#         if num_delete > 0:
#             self.acc_ts = self.acc_ts[num_delete:]
#             self.acc_ts_diff = self.acc_ts_diff[num_delete:]
#             self.acc_num_frames -= num_delete

#     def get(self):
#         if len(self.acc_ts_diff) > config.BATCH_SIZE:
#             # перерасчет фпс
#             self.value = self.acc_num_frames / sum(self.acc_ts_diff)
#         return self.value


# def pid_exists(pid, script_re):
#     try:
#         p = ps.Process(pid)
#         # процесс существует, не зомби и это именно send_data.py -с camera_id
#         send_data_exe = re.compile(script_re)
#         return p.status() != 'zombie' and send_data_exe.search(' '.join(p.cmdline()))
#     except ps.NoSuchProcess:
#         return False
#     except ps.AccessDenied:
#         return False
#     except:
#         traceback.print_exc()
#         return False
