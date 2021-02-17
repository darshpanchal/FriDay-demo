import numpy as np
import sounddevice as sd
import base64
import requests
import time
import json

while (True):
    src = sd.rec(frames=16000,samplerate=16000, channels=1, dtype='float32', blocking=True)
    src = src.reshape(16000,)
    src = np.array(src, dtype='float32')
    src = base64.b64encode(src)
    src = src.decode('utf-8')
    data = {"audio": src}
    # response = requests.post("http://fridaydemo.herokuapp.com/wakeword", json= data)
    response = requests.post("http://127.0.0.1:5000/wakeword")
    print(json.loads(response.content))
    time.sleep(1)