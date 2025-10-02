from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from obamify import obamify

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Fixed target image
target_img = cv2.imread("target.png")

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on('draw_data')
def handle_draw(data):
    try:
        # Decode base64 image
        header, encoded = data.split(',', 1)
        img_bytes = np.frombuffer(base64.b64decode(encoded), np.uint8)
        source_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # Morph with target
        output = obamify(source_img, target_img, alpha=0.6)

        # Encode back to base64
        _, buffer = cv2.imencode('.png', output)
        encoded_output = base64.b64encode(buffer).decode('utf-8')
        emit('update_image', f"data:image/png;base64,{encoded_output}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    # Use eventlet server
    socketio.run(app, host="0.0.0.0", port=5000)
