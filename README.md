# A demo of wakeword detection using TensorFlow

### Dependencies:
- Flask==1.1.2
- gunicorn==20.0.4
- numpy==1.20.1
- scipy==1.4.1 
- tflite_runtime==2.5.0
- Get mel_features.py at this [link](https://github.com/google-coral/project-keyword-spotter/blob/master/mel_features.py).

### Usage:

To run locally, (You will first need to train a model of wakeword detection.)

> python app.py

Or deploy it somewhere (I used Heroku).
You can test api at http://fridaydemo.herokuapp.com 