#!/usr/bin/env python
from flask import Flask, Response, render_template
from camera import Camera
import cv2
from face_detector import FaceDetector
import faiss
import pickle
import numpy as np

app = Flask(__name__)


fd = FaceDetector()

index = faiss.IndexFlatL2(512)
reversed_index = []

with open('features.pkl', 'rb') as f:
    features = pickle.load(f)
    
for user in features:
    for feature in features[user]:
        reversed_index.append(user)
        index.add(np.array([feature]))

def gen(camera):
    global frame
    """Video streaming generator function."""
    idx = 0
    boxes = []
    while True:
        # if idx % 10 != 0:
        #     continue
        try:
            frame = camera.get_frame()
            
            _boxes, features = fd.extract(frame)
            
            if type(_boxes) != type(None):
                boxes = _boxes
            else:
                boxes = []
            
            for box in boxes:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                
                if features != None:
                    D, I = index.search(features.detach().numpy(), 1)
                    cv2.putText(frame, reversed_index[I[0][0]], (int(box[0]), int(box[1])), 0, 5e-3 * 200, (0,255,0), 2)
            
            x = cv2.imencode('.jpg', frame)[1].tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + x + b'\r\n')
        except Exception as e:
            print(e)


@app.route('/')
def live():
    return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
