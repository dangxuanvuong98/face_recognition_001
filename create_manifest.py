import os
import math
import json

root = './.local/data/VN-celeb'
dirs = os.listdir(root)
dirs = sorted(dirs, key = lambda x : len(x) * 10000000000 + int(x))

print(dirs)

n = len(dirs)

n_train = int(math.floor(0.9 * n))

train_manifest = []
test_manifest = []

for i in range(n_train):
    path = os.path.join(root, dirs[i])
    images = os.listdir(path)
    images = sorted(images, key = lambda x : len(x) * 10000000000 + int(x.split('.')[0]))
    n_images = len(images)
    n_images_train = int(math.floor(0.9 * n_images))
    for j in range(n_images_train):
        img = images[j]
        img_path = os.path.join(path, img)
        train_manifest.append({
            'image_path': img_path,
            "class_name": dirs[i],
        })
    
    for j in range(n_images_train, n_images):
        img = images[j]
        img_path = os.path.join(path, img)
        test_manifest.append({
            'image_path': img_path,
            'class_name': dirs[i]
        })

for i in range(n_train, n):
    path = os.path.join(root, dirs[i])
    images = os.listdir(path)
    n_images = len(images)
    for j in range(n_images):
        img = images[j]
        img_path = os.path.join(path, img)
        train_manifest.append({
            'image_path': img_path,
            'class_name': dirs[i]
        })
        
with open('./.local/data/train.json', 'w') as f:
    f.write(json.dumps(train_manifest, indent=4))

with open('./.local/data/test.json', 'w') as f:
    f.write(json.dumps(test_manifest, indent=4))
    
print(len(train_manifest))
print(len(test_manifest))