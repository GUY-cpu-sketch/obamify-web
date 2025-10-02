import asyncio
import base64
import cv2
import numpy as np
from obamify import obamify
import websockets
from flask import Flask, render_template
from threading import Thread

app = Flask(__name__)

# Serve the website
@app.route("/")
def index():
    return render_template("index.html")

# Run Flask in a separate thread
def run_flask():
    app.run(host="0.0.0.0", port=8000)

Thread(target=run_flask).start()

# WebSocket server for real-time Obamify
async def obamify_ws(websocket, path):
    async for message in websocket:
        header, data = message.split(',', 1)
        img_bytes = base64.b64decode(data)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Transform with Obamify
        output = obamify(img)

        _, buffer = cv2.imencode('.png', output)
        encoded = base64.b64encode(buffer).decode('utf-8')
        await websocket.send(f"data:image/png;base64,{encoded}")

start_server = websockets.serve(obamify_ws, "0.0.0.0", 5000)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
