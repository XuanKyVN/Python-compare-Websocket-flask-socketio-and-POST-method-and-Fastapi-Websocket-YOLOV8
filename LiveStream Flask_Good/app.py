from flask import Flask, render_template,Response
import cv2
from ultralytics import YOLO
app = Flask(__name__)
#cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap =cv2.VideoCapture(r"C:\Users\Admin\PythonLession\pic\Traffic1.mp4")

'''
camera = cv2.VideoCapture(0)

for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)

'''

def generate():
     while True:
          ret, frame = cap.read()
          if ret:
               # Them vao boi KY CONG
               model = YOLO(r"C:\Users\Admin\PythonLession\yolo_dataset\yolov8n.pt")
               result = model.predict(frame, device ="cuda")
               frame =result[0].plot()
               #----------------------------------
               (flag, encodedImage) = cv2.imencode(".jpg", frame)
               if not flag:
                    continue
               yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')


          else:
               cap.release()
               break
@app.route("/")
def index():
     return render_template("index.html")
@app.route("/video_feed")
def video_feed():
     return Response(generate(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")
if __name__ == "__main__":
     app.run(host='0.0.0.0', debug=False, threaded=True)
