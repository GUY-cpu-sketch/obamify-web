from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
from obamify import obamify  # make sure the Obamify repo files are in the same folder

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/obamify", methods=["POST"])
def obamify_route():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run Obamify transformation
    output = obamify(img)
    cv2.imwrite("result.png", output)

    return send_file("result.png", mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
