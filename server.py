from flask import Flask, render_template
import asyncio
import base64
import cv2
import numpy as np
from obamify import obamify
import websockets
from threading import Thread

app = Flask(__name__)

# Default target image
target_img = cv2.imread("obama.png")  # You can put any default image here

@app.route("/")
def index():
    return render_template("index.html")

# Flask server in a thread
def run_flask():
    app.run(host="0.0.0.0", port=8000)

Thread(target=run_flask).start()

# WebSocket server
async def obamify_ws(websocket, path):
    global target_img
    async for message in websocket:
        try:
            # message format: data:image/png;base64,xxxx
            header, data = message.split(',', 1)
            img_bytes = base64.b64decode(data)
            source_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # Generate morphed image
            output = obamify(source_img, target_img, alpha=0.6)
            
            # Encode and send back
            _, buffer = cv2.imencode('.png', output)
            encoded = base64.b64encode(buffer).decode('utf-8')
            await websocket.send(f"data:image/png;base64,{encoded}")
        except Exception as e:
            print("Error processing image:", e)

start_server = websockets.serve(obamify_ws, "0.0.0.0", 5000)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
