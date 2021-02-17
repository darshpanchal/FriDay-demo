from flask import Flask, render_template, request, session, jsonify
import numpy as np 
import tflite_runtime.interpreter as tflite
import mel_features
import base64

app = Flask(__name__)

app.config['SECRET_KEY'] = '*insert random generated string*'

interpreter = tflite.Interpreter('model_quant.tflite') # pass your model name here.
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
scale, zero_point = output_details[0]['quantization']

input_shape = input_details[0]['shape']

def getmelspectrogram(src):
    spectrogram = 30 * (mel_features.log_mel_spectrogram(src, audio_sample_rate=16000, log_offset=0.001, window_length_secs = 0.025, hop_length_secs = 0.010, num_mel_bins = 32, lower_edge_hertz=60,
            upper_edge_hertz=3800) - np.log(1e-3))
    spectrogram = np.array(np.ceil(spectrogram), dtype=np.uint8)
    return spectrogram

def getoutput(src):
    src = np.asarray(src)
    src = src.reshape(16000,)
    melspec = getmelspectrogram(src)
    melspec = np.array(melspec, dtype=np.uint8).reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], melspec)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output = scale * (output_data - zero_point)
    return output

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("homepage.html")
    else:
        return "POST not accepted.."

@app.route("/wakeword", methods=["POST", "GET"])
def wakeword():
    if request.method == "POST":
        data = request.get_json()
        src = data['audio'].encode()
        src = base64.decodebytes(src)
        src = np.frombuffer(src, dtype='float32')
        result = getoutput(src)
        return jsonify({"result": str(result[0][0])})
    else:
        return "GET not accepted.."


if __name__ == "__main__":
    app.run(debug=True)
