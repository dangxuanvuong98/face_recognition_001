from face_detector import FaceDetector
import pickle
import os
from PIL import Image

def main():
    features = dict()
    users_dir = './users'
    subdirs = os.listdir(users_dir)
    for dir in subdirs:
        fd = FaceDetector()
        features[dir] = []
        path = os.path.join(users_dir, dir)
        images = os.listdir(path)
        for image in images:
            image_path = os.path.join(path, image)
            img = Image.open(image_path)
            _, ft = fd.extract(img)
            features[dir].append(ft[0].detach().numpy())
    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)

if __name__ == '__main__':
    main()