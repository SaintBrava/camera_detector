import os
# from dataclasses import dataclass
from distutils.util import strtobool
from pathlib import Path

# from dotenv import load_dotenv

# env_filename = os.environ.get("PRODUCER_CONFIG", "default.env")
# env_path = Path(__file__).parent / "configs" / env_filename
# # load_dotenv(dotenv_path=env_path)

# BATCH_SIZE = int(os.getenv("BATCH_SIZE", 12))
# EXPECTED_BATCH_TIME = float(os.getenv("EXPECTED_BATCH_TIME", 1.6))  # in seconds

# # какие сетки/эвристики включить (на случай, если в бд нет этих настроек) + пути до файлов сеток
# YOLOV4_WEIGHTS = os.path.expanduser(os.getenv("YOLOV4_WEIGHTS"))
# OCR = strtobool(os.getenv("OCR"))

# YOLOV5GRN_WEIGHTS = os.path.expanduser(os.getenv("YOLOV5GRN_WEIGHTS"))
# TEXT_DETECTOR_WEIGHTS = os.path.expanduser(os.getenv("TEXT_DETECTOR_WEIGHTS"))
# OPT_DETECTOR = strtobool(os.environ.get("OPT_DETECTOR", "False"))  # необязательно
# OPT_DETECTOR_WEIGHTS = os.path.expanduser(os.environ.get("OPT_DETECTOR_WEIGHTS", "latest"))

# TEXT_DETECTOR = OCR
# DTP = strtobool(os.getenv("DTP"))
# YOLOV5DTP_WEIGHTS = os.path.expanduser(os.getenv("YOLOV5DTP_WEIGHTS"))

# # эвристики
# PERES_POLOS = strtobool(os.getenv("PERES_POLOS", "False"))
# ZAPRET_ZONE = strtobool(os.getenv("ZAPRET_ZONE", "False"))
# TRAFFIC_ON_ROAD = strtobool(os.getenv("TRAFFIC_ON_ROAD", "False"))
# DIRECTION = strtobool(os.getenv("DIRECTION", "False"))
# SPEED = strtobool(os.getenv("SPEED", "False"))
# CAMERA_SHIFT = strtobool(os.getenv("CAMERA_SHIFT", "False"))
# SPEED_CHANGE = strtobool(os.getenv("SPEED_CHANGE", "False"))
# DIRECTION_CHANGE = strtobool(os.getenv("DIRECTION_CHANGE", "False"))
# FROM_ROAD = strtobool(os.getenv("FROM_ROAD", "False"))
# DETECT_PEOPLE = strtobool(os.getenv("DETECT_PEOPLE", "False"))

# AGGREGATE = strtobool(os.getenv("AGGREGATE", "True"))
# if AGGREGATE:
#     AGGREGATION_INTERVAL = int(os.environ.get("AGGREGATION_INTERVAL", 60))
# else:
#     AGGREGATION_INTERVAL = None

# DEBUG = strtobool(os.environ.get("DEBUG", "False"))
# DEBUG_MEM = strtobool(os.environ.get("DEBUG_MEM", "False"))
# SAVE_FRAMES = strtobool(os.getenv("SAVE_FRAMES", "False"))
# if SAVE_FRAMES:
#     SAVE_INTERVAL = float(os.getenv("SAVE_INTERVAL", 60))
# else:
#     SAVE_INTERVAL = None

# MONITORING_INTERVAL = int(os.environ.get("MONITORING_INTERVAL", 60))

# URL_PREFIX = os.environ.get("URL_PREFIX")

# # костыль
# MAX_SPEED = float(os.environ.get("MAX_SPEED", 55))
# DEFAULT_FPS = float(os.environ.get("DEFAULT_FPS", 25))
# MIN_AREA = float(os.environ.get("MIN_AREA", 0.005))
# # эвристика, минимальная скорость для определения направления тс
# DIRECTION_MIN_SPEED = int(os.environ.get('DIRECTION_MIN_SPEED', 0))

# # настройки выделения памяти на гпу под сетки (заранее заданный лимит или динамическое выделение)
# YOLOV5GRN_MEMORY_LIMIT = int(os.environ.get("YOLOV5GRN_MEMORY_LIMIT", 0))
# YOLOV5DTP_MEMORY_LIMIT = int(os.environ.get("YOLOV5DTP_MEMORY_LIMIT", 0))
# TEXT_DETECTOR_MEMORY_LIMIT = int(os.environ.get("TEXT_DETECTOR_MEMORY_LIMIT", 256))
# OPT_DETECTOR_MEMORY_LIMIT = int(os.environ.get("OPT_DETECTOR_MEMORY_LIMIT", 0))
# YOLOV4_MEMORY_LIMIT = int(os.environ.get("YOLOV4_MEMORY_LIMIT", 1024))

# # максимальный % занятой видеопамяти GPU, для запуска на нем анализа видео
# MEM_UTIL_MAX = float(os.environ.get("MEM_UTIL_MAX", 1.0))
# # максимальный % загруженности GPU, для запуска на нем анализа видео
# GPU_UTIL_MAX = float(os.environ.get("GPU_UTIL_MAX", 1.0))
# GPU_UTIL_PER_PROCESS = 100.0 - GPU_UTIL_MAX

# USE_MEMORY_LIMITS = strtobool(os.environ.get('USE_MEMORY_LIMITS', 'False'))

# # какой библиотекой читать кадры
# VIDEO_CAPTURE = os.environ.get('VIDEO_CAPTURE', 'CV2')  # FFMPEG || CV2
# # откуда получать кадры
# VIDEO_PROVIDER = os.getenv('VIDEO_PROVIDER', 'CV2')     # FFMPEG || CV2 || REDIS

# kafka producer settings
KAFKA_SEND_DATA_ENABLE = strtobool(os.environ.get('KAFKA_SEND_DATA_ENABLE', 'true'))
DEBUG_SEND_DATA_ENABLE = strtobool(os.environ.get('DEBUG_SEND_DATA_ENABLE', 'false'))
KAFKA_SERVER = os.environ.get('KAFKA_SERVER', '127.0.0.1:9092')
KAFKA_BATCH_SIZE = int(os.environ.get('KAFKA_BATCH_SIZE', 16384))
KAFKA_LINGER_MS = int(os.environ.get('KAFKA_LINGER_MS', 0))
KAFKA_TOPIC = os.environ.get('KAFKA_TOPIC', 'report')
KAFKA_TOPIC_MONITORING = os.environ.get('KAFKA_TOPIC_MONITORING', 'monitoring')
KAFKA_TOPIC_AGGREGATION = os.environ.get('KAFKA_TOPIC_AGGREGATION', 'aggregation')

# # redis settings
# REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
# REDIS_PORT = os.getenv("REDIS_PORT", "6379")
# REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# VIDEO_FILES_PATH = Path(os.environ.get('VIDEO_FILES_PATH', '/mnt/6tb-hdd/videos/'))
# VIDEO_FILES_PATH.mkdir(exist_ok=True, parents=True)

# VIDEO_FRAMES_PATH = Path(os.environ.get('VIDEO_FRAMES_PATH', '/mnt/6tb-hdd/cameras/'))
# VIDEO_FRAMES_PATH.mkdir(exist_ok=True, parents=True)

# SAVE_ORIG_FRAMES = strtobool(os.getenv("SAVE_ORIG_FRAMES", "False"))
# SAVE_ORIG_INTERVAL = float(os.getenv("SAVE_ORIG_INTERVAL", 5))
# ORIG_VIDEO_FRAMES_PATH = os.getenv('ORIG_VIDEO_FRAMES_PATH')
# if ORIG_VIDEO_FRAMES_PATH:
#     ORIG_VIDEO_FRAMES_PATH = Path(ORIG_VIDEO_FRAMES_PATH)
#     ORIG_VIDEO_FRAMES_PATH.mkdir(exist_ok=True, parents=True)

# SAVE_COLOR_PATH = os.getenv('SAVE_COLOR_PATH')
# if SAVE_COLOR_PATH:
#     SAVE_COLOR_PATH = Path(SAVE_COLOR_PATH)
#     SAVE_COLOR_PATH.mkdir(exist_ok=True, parents=True)

# SAVE_MODEL_PATH = os.getenv('SAVE_MODEL_PATH')
# if SAVE_MODEL_PATH:
#     SAVE_MODEL_PATH = Path(SAVE_MODEL_PATH)
#     SAVE_MODEL_PATH.mkdir(exist_ok=True, parents=True)

LOGS_PATH = os.environ.get("LOGS_PATH", "./logs")
Path(LOGS_PATH).mkdir(parents=True, exist_ok=True)
REPORTS_PATH = Path(os.environ.get("REPORTS_PATH", "./reports"))
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


# @dataclass
# class DbConfig:
#     password: str
#     user: str
#     db: str
#     host: str = '127.0.0.1'
#     port: str = '5432'

#     @property
#     def dsn(self):
#         return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


# # GUI
# GUI_POSTGRES_CONFIG = DbConfig(
#     password=os.getenv('GUI_POSTGRES_PASSWORD'),
#     user=os.getenv('GUI_POSTGRES_USER'),
#     db=os.getenv('GUI_POSTGRES_DB'),
#     host=os.getenv('GUI_POSTGRES_HOST'),
#     port=os.getenv('GUI_POSTGRES_PORT')
# )
# # Consumer
# CON_POSTGRES_CONFIG = DbConfig(
#     password=os.getenv('CON_POSTGRES_PASSWORD'),
#     user=os.getenv('CON_POSTGRES_USER'),
#     db=os.getenv('CON_POSTGRES_DB'),
#     host=os.getenv('CON_POSTGRES_HOST'),
#     port=os.getenv('CON_POSTGRES_PORT')
# )
# # Operation Log
# OPERATION_POSTGRES_CONFIG = DbConfig(
#     password=os.getenv('OPERATION_POSTGRES_PASSWORD'),
#     user=os.getenv('OPERATION_POSTGRES_USER'),
#     db=os.getenv('OPERATION_POSTGRES_DB'),
#     host=os.getenv('OPERATION_POSTGRES_HOST'),
#     port=os.getenv('OPERATION_POSTGRES_PORT')
# )

# GUI_HOST = os.environ.get("GUI_HOST", "192.168.88.20:5000")
# APP_HOST = os.environ.get("APP_HOST", "127.0.0.1:5000")

# TRANSPORT_CLASSES = {'car', 'truck', 'bus', 'motorbike'}

# # thresholds
# PERES_POLOS_THRESHOLD = float(os.environ.get("PERES_POLOS_THRESHOLD", 0.0))
# ZAPRET_ZONE_THRESHOLD = float(os.environ.get("ZAPRET_ZONE_THRESHOLD", 0.5))
# STOPPING_THRESHOLD = float(os.environ.get("STOPPING_THRESHOLD", 0.5))
# # thresholds (db fallback)
# DTP_THRESHOLD = float(os.environ.get("DTP_THRESHOLD", 0.9))
# COLOR_THRESHOLD = float(os.environ.get("COLOR_THRESHOLD", 0.5))
# MAKER_THRESHOLD = float(os.environ.get("MAKER_THRESHOLD", 0.0))

# # bbox min sizes for detection
# MIN_HEIGHT_BBOX = int(os.environ.get("MIN_HEIGHT_BBOX", 0))
# MIN_WIDTH_BBOX = int(os.environ.get("MIN_WIDTH_BBOX", 0))

# # bbox visualisation
# FONT_FAMILY = os.environ.get("FONT_FAMILY", "consola.ttf")
# FONT_SIZE = int(os.environ.get("FONT_SIZE", 20))
# FONT_FILL = (0, 0, 0)  # цвет текста
# BBOX_WIDTH = int(os.environ.get("BBOX_WIDTH", 3))
# BBOX_COLOR = (0, 255, 0)
# BBOX_COLOR_ACCIDENT = (255, 0, 0)
# SHORT = strtobool(os.environ.get("SHORT", "True"))
