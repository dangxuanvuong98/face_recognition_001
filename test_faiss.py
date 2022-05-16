#!/usr/bin/env python
from face_detector import FaceDetector
import faiss
import pickle
from PIL import Image
import numpy as np

fd = FaceDetector()

index = faiss.IndexFlatL2(512)
reversed_index = []

with open('features.pkl', 'rb') as f:
    features = pickle.load(f)
    
for user in features:
    for feature in features[user]:
        reversed_index.append(user)
        index.add(np.array([feature]))
        
img = Image.open('./users/vuongdx/test.jpg')
_boxes, features = fd.extract(img)
        
D, I = index.search(features.detach().numpy(), 5)
print(D)
print(I)