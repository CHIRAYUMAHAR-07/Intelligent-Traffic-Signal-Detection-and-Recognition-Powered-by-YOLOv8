Access/Enable GPU for fast processing
import os
if 'COLAB_GPU' in os.environ:
print("GPU is enabled !")
else:
print("GPU is not enabled!")
GPU is enabled !
INSTALL The YOLO-8
!pip install ultralytics==8.0.2
Requirement already satisfied: ultralytics==8.0.2 in /usr/local/lib/python3.11/dist-packages (8.0.2)
Requirement already satisfied: hydra-core>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (1.3.2)
Requirement already satisfied: matplotlib>=3.2.2 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (3.10.0)
Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (1.26.4)
Requirement already satisfied: opencv-python>=4.1.1 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (4.10.0.84)
Requirement already satisfied: Pillow>=7.1.2 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (11.1.0)
Requirement already satisfied: PyYAML>=5.3.1 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (6.0.2)
Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (2.32.3)
Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (1.13.1)
Requirement already satisfied: torch>=1.7.0 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (2.5.1+cu121)
Requirement already satisfied: torchvision>=0.8.1 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (0.20.1+cu121)
Requirement already satisfied: tqdm>=4.64.0 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (4.67.1)
Requirement already satisfied: tensorboard>=2.4.1 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (2.17.1)
Requirement already satisfied: pandas>=1.1.4 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (2.2.2)
Requirement already satisfied: seaborn>=0.11.0 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (0.13.2)
Requirement already satisfied: ipython in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (7.34.0)
Requirement already satisfied: psutil in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (5.9.5)
Requirement already satisfied: thop>=0.1.1 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (0.1.1.post2209072238
Requirement already satisfied: GitPython>=3.1.24 in /usr/local/lib/python3.11/dist-packages (from ultralytics==8.0.2) (3.1.44)
Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.11/dist-packages (from GitPython>=3.1.24->ultralytics==8.0.2
Requirement already satisfied: omegaconf<2.4,>=2.2 in /usr/local/lib/python3.11/dist-packages (from hydra-core>=1.2.0->ultralytics==8
Requirement already satisfied: antlr4-python3-runtime==4.9.* in /usr/local/lib/python3.11/dist-packages (from hydra-core>=1.2.0->ultr
Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from hydra-core>=1.2.0->ultralytics==8.0.2) (24.
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.2.2->ultralytics==8.0.
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.2.2->ultralytics==8.0.2) (
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.2.2->ultralytics==8.0
Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.2.2->ultralytics==8.0
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.2.2->ultralytics==8.0.
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.2.2->ultralytics==
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.1.4->ultralytics==8.0.2) (2024
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.1.4->ultralytics==8.0.2) (20
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests>=2.23.0->ultralytic
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests>=2.23.0->ultralytics==8.0.2) (3
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests>=2.23.0->ultralytics==8.0
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests>=2.23.0->ultralytics==8.0
Requirement already satisfied: absl-py>=0.4 in /usr/local/lib/python3.11/dist-packages (from tensorboard>=2.4.1->ultralytics==8.0.2)
Requirement already satisfied: grpcio>=1.48.2 in /usr/local/lib/python3.11/dist-packages (from tensorboard>=2.4.1->ultralytics==8.0.2
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.11/dist-packages (from tensorboard>=2.4.1->ultralytics==8.0.
Requirement already satisfied: protobuf!=4.24.0,>=3.19.6 in /usr/local/lib/python3.11/dist-packages (from tensorboard>=2.4.1->ultraly
Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.11/dist-packages (from tensorboard>=2.4.1->ultralytics==8
Requirement already satisfied: six>1.9 in /usr/local/lib/python3.11/dist-packages (from tensorboard>=2.4.1->ultralytics==8.0.2) (1.17
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.11/dist-packages (from tensorboard>=2.
Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from tensorboard>=2.4.1->ultralytics==8.0.
Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultralytics==8.0.2) (3.16.1)
Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultralytics==8
Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultralytics==8.0.2) (3.4.2)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultralytics==8.0.2) (3.1.5)
Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultralytics==8.0.2) (2024.10.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultral
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.1.105 in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultr
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultral
Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultralytics
Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultralytic
Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultralytic
Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultralyt
Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultral
Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in /usr/local/lib/python3.11/dist-packages (from torch>=1.7.0->ultral
import ultralytics
ultralytics.checks()
Ultralytics YOLOv8.0.2 🚀 Python-3.11.11 torch-2.5.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
Setup complete ✅ (2 CPUs, 12.7 GB RAM, 31.1/112.6 GB disk)
/content''
%pwd
!mkdir TrafficLightDetection
!ls
TrafficLightDetection
path = "/content/TrafficLightDetection"
/content''
%pwd
/content/TrafficLightDetection''
import os
os.chdir("/content/TrafficLightDetection")
%pwd
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="HEdknxxKTOdanD5UL3gR")
project = rf.workspace("sohryu").project("traffic-light-detector-ncnsz")
version = project.version(2)
dataset = version.download("yolov8")
Collecting roboflow
Downloading roboflow-1.1.50-py3-none-any.whl.metadata (9.7 kB)
Requirement already satisfied: certifi in /usr/local/lib/python3.11/dist-packages (from roboflow) (2024.12.14)
Collecting idna==3.7 (from roboflow)
Downloading idna-3.7-py3-none-any.whl.metadata (9.9 kB)
Requirement already satisfied: cycler in /usr/local/lib/python3.11/dist-packages (from roboflow) (0.12.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from roboflow) (1.4.8)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.11/dist-packages (from roboflow) (3.10.0)
Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.11/dist-packages (from roboflow) (1.26.4)
Requirement already satisfied: opencv-python-headless==4.10.0.84 in /usr/local/lib/python3.11/dist-packages (from roboflow) (4.10.0.84)
Requirement already satisfied: Pillow>=7.1.2 in /usr/local/lib/python3.11/dist-packages (from roboflow) (11.1.0)
Requirement already satisfied: python-dateutil in /usr/local/lib/python3.11/dist-packages (from roboflow) (2.8.2)
Collecting python-dotenv (from roboflow)
Downloading python_dotenv-1.0.1-py3-none-any.whl.metadata (23 kB)
Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from roboflow) (2.32.3)
Requirement already satisfied: six in /usr/local/lib/python3.11/dist-packages (from roboflow) (1.17.0)
Requirement already satisfied: urllib3>=1.26.6 in /usr/local/lib/python3.11/dist-packages (from roboflow) (2.3.0)
Requirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.11/dist-packages (from roboflow) (4.67.1)
Requirement already satisfied: PyYAML>=5.3.1 in /usr/local/lib/python3.11/dist-packages (from roboflow) (6.0.2)
Requirement already satisfied: requests-toolbelt in /usr/local/lib/python3.11/dist-packages (from roboflow) (1.0.0)
Collecting filetype (from roboflow)
Downloading filetype-1.2.0-py2.py3-none-any.whl.metadata (6.5 kB)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->roboflow) (1.3.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib->roboflow) (4.55.3)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib->roboflow) (24.2)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->roboflow) (3.2.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->roboflow) (3.4.1)
Downloading roboflow-1.1.50-py3-none-any.whl (81 kB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 81.5/81.5 kB 5.0 MB/s eta 0:00:00
Downloading idna-3.7-py3-none-any.whl (66 kB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.8/66.8 kB 6.3 MB/s eta 0:00:00
Downloading filetype-1.2.0-py2.py3-none-any.whl (19 kB)
Downloading python_dotenv-1.0.1-py3-none-any.whl (19 kB)
Installing collected packages: filetype, python-dotenv, idna, roboflow
Attempting uninstall: idna
Found existing installation: idna 3.10
Uninstalling idna-3.10:
Successfully uninstalled idna-3.10
Successfully installed filetype-1.2.0 idna-3.7 python-dotenv-1.0.1 roboflow-1.1.50
loading Roboflow workspace...
loading Roboflow project...
Downloading Dataset Version Zip in traffic-light-detector-2 to yolov8:: 100%|██████████| 30619/30619 [00:00<00:00, 49859.96it/s]
Extracting Dataset Version Zip to traffic-light-detector-2 in yolov8:: 100%|██████████| 1476/1476 [00:00<00:00, 7869.98it/s]
os.chdir("/content/TrafficLightDetection/traffic-light-detector-2")
!yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=15 imgsz=640 batch=16
Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt to yolov8n.pt...
100% 6.23M/6.23M [00:00<00:00, 103MB/s]
/usr/local/lib/python3.11/dist-packages/ultralytics/nn/tasks.py:341: FutureWarning: You are using `torch.load` with `weights_only=Fal
ckpt = torch.load(attempt_download(weight), map_location='cpu') # load
yolo/engine/trainer: task=detect, mode=train, model=yolov8n.pt, data=data.yaml, epochs=15, patience=50, batch=16, imgsz=640, save=Tru
Ultralytics YOLOv8.0.2 🚀 Python-3.11.11 torch-2.5.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
/usr/local/lib/python3.11/dist-packages/ultralytics/yolo/engine/trainer.py:118: FutureWarning: `torch.cuda.amp.GradScaler(args...)` i
self.scaler = amp.GradScaler(enabled=self.amp)
Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
100% 755k/755k [00:00<00:00, 22.9MB/s]
2025-01-16 10:56:34.595656: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempti
2025-01-16 10:56:34.616007: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempt
2025-01-16 10:56:34.622788: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attem
2025-01-16 10:56:34.638257: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-01-16 10:56:36.023044: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Overriding model.yaml nc=80 with nc=3
from n params module arguments
0 -1 1 464 ultralytics.nn.modules.Conv [3, 16, 3, 2]
1 -1 1 4672 ultralytics.nn.modules.Conv [16, 32, 3, 2]
2 -1 1 7360 ultralytics.nn.modules.C2f [32, 32, 1, True]
3 -1 1 18560 ultralytics.nn.modules.Conv [32, 64, 3, 2]
4 -1 2 49664 ultralytics.nn.modules.C2f [64, 64, 2, True]
5 -1 1 73984 ultralytics.nn.modules.Conv [64, 128, 3, 2]
6 -1 2 197632 ultralytics.nn.modules.C2f [128, 128, 2, True]
7 -1 1 295424 ultralytics.nn.modules.Conv [128, 256, 3, 2]
8 -1 1 460288 ultralytics.nn.modules.C2f [256, 256, 1, True]
9 -1 1 164608 ultralytics.nn.modules.SPPF [256, 256, 5]
10 -1 1 0 torch.nn.modules.upsampling.Upsample [None, 2, 'nearest']
11 [-1, 6] 1 0 ultralytics.nn.modules.Concat [1]
12 -1 1 148224 ultralytics.nn.modules.C2f [384, 128, 1]
13 -1 1 0 torch.nn.modules.upsampling.Upsample [None, 2, 'nearest']
14 [-1, 4] 1 0 ultralytics.nn.modules.Concat [1]
15 -1 1 37248 ultralytics.nn.modules.C2f [192, 64, 1]
16 -1 1 36992 ultralytics.nn.modules.Conv [64, 64, 3, 2]
17 [-1, 12] 1 0 ultralytics.nn.modules.Concat [1]
18 -1 1 123648 ultralytics.nn.modules.C2f [192, 128, 1]
19 -1 1 147712 ultralytics.nn.modules.Conv [128, 128, 3, 2]
20 [-1, 9] 1 0 ultralytics.nn.modules.Concat [1]
21 -1 1 493056 ultralytics.nn.modules.C2f [384, 256, 1]
22 [15, 18, 21] 1 751897 ultralytics.nn.modules.Detect [3, [64, 128, 256]]
Model summary: 225 layers, 3011433 parameters, 3011417 gradients, 8.2 GFLOPs
Transferred 319/355 items from pretrained weights
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias
train: Scanning /content/TrafficLightDetection/traffic-light-detector-2/train/labels... 642 images, 33 backgrounds, 0 corrupt: 100% 6
Signal received. 15 <frame at 0xf9133e674c0, file '/usr/lib/python3.11/threading.py', line 839, code <listcomp>>
train: New cache created: /content/TrafficLightDetection/traffic-light-detector-2/train/labels.cache
/usr/local/lib/python3.11/dist-packages/albumentations/__init__.py:24: UserWarning: A new version of Albumentations is available: 2.0
check_for_updates()
/usr/local/lib/python3.11/dist-packages/albumentations/core/composition.py:205: UserWarning: Got processor for bboxes, but no transfo
self._set_keys()
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, num_output_channels=3, method=
val: Scanning /content/TrafficLightDetection/traffic-light-detector-2/valid/labels... 45 images, 4 backgrounds, 0 corrupt: 100% 45/45
Signal received. 15 <frame at 0xf913240aea0, file '/usr/lib/python3.11/weakref.py', line 462, code items>
!ls '/content/TrafficLightDetection/traffic-light-detector-2/runs/detect/train'
args.yaml F1_curve.png predictions.json results.png
confusion_matrix.png P_curve.png R_curve.png weights
events.out.tfevents.1737024997.7954e381f15d.7559.0 PR_curve.png results.csv
from IPython.display import Image, display
RESULT_DATA = "/content/TrafficLightDetection/traffic-light-detector-2/runs/detect/train"
image_path = f"{RESULT_DATA}/confusion_matrix.png"
# Display the image
display(Image(filename=image_path)
![image](https://github.com/user-attachments/assets/20a9d6d0-f7a6-4b98-afe4-aa4515c785c5)
from IPython.display import Image, display
RESULT_DATA = "/content/TrafficLightDetection/traffic-light-detector-2/runs/detect/train"
image_path = f"{RESULT_DATA}/F1_curve.png"
# Display the image
display(Image(filename=image_path))
![image](https://github.com/user-attachments/assets/d6e05846-98e5-4063-b374-8187b1ad628e)
from IPython.display import Image, display
RESULT_DATA = "/content/TrafficLightDetection/traffic-light-detector-2/runs/detect/train"
image_path = f"{RESULT_DATA}/P_curve.png"
# Display the image
display(Image(filename=image_path))
![image](https://github.com/user-attachments/assets/69c4dd09-022d-445b-bfde-df48355d0a09)
from IPython.display import Image, display
RESULT_DATA = "/content/TrafficLightDetection/traffic-light-detector-2/runs/detect/train"
image_path = f"{RESULT_DATA}/PR_curve.png"
# Display the image
display(Image(filename=image_path))
![image](https://github.com/user-attachments/assets/bdd8d708-140b-449a-b836-3f720b33a301)
from IPython.display import Image, display
RESULT_DATA = "/content/TrafficLightDetection/traffic-light-detector-2/runs/detect/train"
image_path = f"{RESULT_DATA}/R_curve.png"
# Display the image
display(Image(filename=image_path))
![image](https://github.com/user-attachments/assets/ec8bcaf9-97a6-4008-9e71-2f1f2ba2832a)
from IPython.display import Image, display
RESULT_DATA = "/content/TrafficLightDetection/traffic-light-detector-2/runs/detect/train"
image_path = f"{RESULT_DATA}/results.png"
# Display the image
display(Image(filename=image_path))
![image](https://github.com/user-attachments/assets/7de77b2b-2bcd-4ca5-9cbe-ad04df783c93)
from IPython.display import Image, display
RESULT_DATA = "/content/TrafficLightDetection/traffic-light-detector-2/runs/detect/train"
image_path = f"{RESULT_DATA}/results.png"
# Display the image
display(Image(filename=image_path))
![image](https://github.com/user-attachments/assets/4306a8bf-cbfd-486e-b3fe-822d4a286e53)
from IPython.display import Image, display
RESULT_DATA = "/content/TrafficLightDetection/traffic-light-detector-2/runs/detect/train"
image_path = f"{RESULT_DATA}/results.png"
# Display the image
display(Image(filename=image_path))
import pandas as pd
RESULT_DATA = "/content/TrafficLightDetection/traffic-light-detector-2/runs/detect/train"
csv_file_path = f'{RESULT_DATA}/results.csv'
df = pd.read_csv(csv_file_path)
# Print the first 5 rows of the DataFrame
print(df.head(5))
epoch train/box_loss train/cls_loss \
0 0 1.4748 3.5993
1 1 1.5106 2.7417
2 2 1.4897 2.1910
3 3 1.4962 1.9980
4 4 1.4463 1.8453
train/dfl_loss metrics/precision(B) ... val/cls_loss \
0 1.5144 0.00352 ... inf
1 1.4627 0.34835 ... inf
2 1.4175 0.66816 ... inf
3 1.4585 0.75202 ... inf
4 1.4048 0.65002 ... inf
val/dfl_loss lr/pg0 lr/pg1 \
0 0 0.070732 0.003252
1 0 0.040297 0.006151
2 0 0.009422 0.008609
3 0 0.008020 0.008020
4 0 0.008020 0.008020
lr/pg2
0 0.003252
1 0.006151
2 0.008609
3 0.008020
4 0.008020
[5 rows x 14 columns]
# Print the first 5 rows of the DataFrame
print(df.head(5))
epoch train/box_loss train/cls_loss \
0 0 1.4748 3.5993
1 1 1.5106 2.7417
2 2 1.4897 2.1910
3 3 1.4962 1.9980
4 4 1.4463 1.8453
train/dfl_loss metrics/precision(B) ... val/cls_loss \
0 1.5144 0.00352 ... inf
1 1.4627 0.34835 ... inf
2 1.4175 0.66816 ... inf
3 1.4585 0.75202 ... inf
4 1.4048 0.65002 ... inf
val/dfl_loss lr/pg0 lr/pg1 \
0 0 0.070732 0.003252
1 0 0.040297 0.006151
2 0 0.009422 0.008609
3 0 0.008020 0.008020
4 0 0.008020 0.008020
lr/pg2
0 0.003252
1 0.006151
2 0.008609
3 0.008020
4 0.008020
[5 rows x 14 columns]
import pandas as pd
RESULT_DATA = "/content/TrafficLightDetection/traffic-light-detector-2/runs/detect/train"
csv_file_path = f'{RESULT_DATA}/results.csv'
df = pd.read_csv(csv_file_path)
# Print the first 5 rows of the DataFrame
print(df.tail(1))
epoch train/box_loss train/cls_loss \
14 14 1.2467 1.2316
train/dfl_loss metrics/precision(B) ... val/cls_loss \
14 1.2777 0.91966 ... inf
val/dfl_loss lr/pg0 lr/pg1 \
14 0 0.00142 0.00142
lr/pg2
14 0.00142
[1 rows x 14 columns]
WEIGHTS_PATH = "/content/TrafficLightDetection/traffic-light-detector-2/runs/detect/train/weights"
!yolo task=detect mode=predict model={WEIGHTS_PATH}/best.pt conf=0.25 source={path}/test/images save=True
2025-01-16 11:18:55.199589: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting
2025-01-16 11:18:55.219178: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting
2025-01-16 11:18:55.225277: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempti
2025-01-16 11:18:55.243279: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CP
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-01-16 11:18:56.798386: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Ultralytics YOLOv8.0.2 🚀 Python-3.11.11 torch-2.5.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
/usr/local/lib/python3.11/dist-packages/ultralytics/nn/tasks.py:303: FutureWarning: You are using `torch.load` with `weights_only=False`
ckpt = torch.load(attempt_download(w), map_location='cpu') # load
Fusing layers...
Model summary: 168 layers, 3006233 parameters, 0 gradients, 8.1 GFLOPs
Error executing job with overrides: ['task=detect', 'mode=predict', 'model=/content/TrafficLightDetection/traffic-light-detector-2/runs/
Traceback (most recent call last):
File "/usr/local/lib/python3.11/dist-packages/ultralytics/yolo/cli.py", line 52, in cli
func(cfg)
File "/usr/local/lib/python3.11/dist-packages/hydra/main.py", line 83, in decorated_main
return task_function(cfg_passthrough)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.11/dist-packages/ultralytics/yolo/v8/detect/predict.py", line 92, in predict
predictor()
File "/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
return func(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.11/dist-packages/ultralytics/yolo/engine/predictor.py", line 152, in __call__
model = self.model if self.done_setup else self.setup(source, model)
^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.11/dist-packages/ultralytics/yolo/engine/predictor.py", line 136, in setup
self.dataset = LoadImages(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=self.args.vid_stride)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.11/dist-packages/ultralytics/yolo/data/dataloaders/stream_loaders.py", line 171, in __init__
raise FileNotFoundError(f'{p} does not exist')
FileNotFoundError: /content/TrafficLightDetection/test/images does not exist
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
/content/TrafficLightDetection/traffic light detector 2/runs/detect/train/weights''
os.chdir("/content")
/content''
%pwd
!mkdir MANUALL_TEST
os.chdir("/content/MANUALL_TEST")






