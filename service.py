#!/usr/bin/env python
import pickle
import sqlite3
from datetime import datetime

import cv2
import faiss
import numpy as np
from flask import Flask, Response, render_template

from camera import Camera
from face_detector import FaceDetector

app = Flask(__name__)


fd = FaceDetector()

index = faiss.IndexFlatL2(512)
reversed_index = []

with open('./.local/data/features.pkl', 'rb') as f:
    features = pickle.load(f)
    
db_con = sqlite3.connect("./.local/data/attendance.db")
cur = db_con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS attendance(student, time)")
db_con.close()
    
for user in features:
    for feature in features[user]:
        reversed_index.append(user)
        index.add(np.array([feature]))
        
threshold = 1.0

def gen(camera):
    global frame
    """Video streaming generator function."""
    idx = 0
    boxes = []
    db_con = sqlite3.connect("./.local/data/attendance.db")
    cur = db_con.cursor()
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
                    text = reversed_index[I[0][0]] if D[0][0] < threshold else 'Unknown'
                    cv2.putText(frame, text, (int(box[0]), int(box[1])), 0, 5e-3 * 200, (0,255,0), 2)
                    
                    if text != 'Unknown':
                        try:
                            query = f"""
                            INSERT INTO attendance VALUES
                            ("{text}", "{datetime.now()}")
                            """
                            cur.execute(
                                query
                            )
                        except Exception as e:
                            print("insert error", e)
            try:
                db_con.commit()
            except Exception as e:
                print("commit error", e)
            
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
