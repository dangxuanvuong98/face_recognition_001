from PIL import Image
from face_detector import FaceDetector
import torch

image_paths = [
    'img/test.jpg', 
    'img/test_2.jpg',
    'img/test_3.jfif',
    'img/test_4.jpg'
]

images = [Image.open(path) for path in image_paths]

fd = FaceDetector()

emb = [fd.extract(img) for img in images]

dis = []
n = len(emb)
for i in range(n):
    dis.append([])
    for j in range(n):
        dis[i].append(torch.linalg.norm(emb[i] - emb[j]).item())
        
print(dis)