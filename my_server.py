# Built-in Lib
import base64
import random
from io import BytesIO

# Own function
from train_model import create_model,IMAGE_SHAPE,MAX_CHAR_NUM,NUM_CHAR_CLASS, make_predictions
from gen_captcha import gen_captcha_image

# External Lib
from flask import Flask, render_template, request, url_for, jsonify
from captcha.image import ImageCaptcha
from PIL import Image
from nltk.corpus import brown
import numpy as np


app = Flask(__name__)

# Load trained model
model = create_model(IMAGE_SHAPE, MAX_CHAR_NUM, NUM_CHAR_CLASS)
model.load_weights('static/weights.384-0.12.hdf5')

# Instantiate image captcha generator
imageCaptcha = ImageCaptcha(width=200, height=80)

# Pick first 1000 words with length 4-6 from corpus brown
words = list(filter(lambda x: len(x)>=4 and len(x)<=6, brown.words(categories=['humor'])))[:1000]

if __name__ == '__main__':
    app.run(debug=True)
	
# Source: http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
# With some modifications to fit into the model
def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(1,img.size[1], img.size[0], 3)

# Rendering image in HTML
# Source: https://www.pythonanywhere.com/forums/topic/5017/
def get_image_bytes(image):
    figfile = BytesIO()
    image.save(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')
    return figdata_png

# Generate a random word from nltk corpus
def generate_random_word():
    return random.choice(words)

@app.route('/generate_and_predict', methods=['GET'])
def generate_and_predict():
    word = generate_random_word()
    image = gen_captcha_image(imageCaptcha, word)
    # Make predictions	
    prediction = make_predictions(model, PIL2array(image))[0]
    # Get image bytes data for rendering
    figdata_png = get_image_bytes(image)
    # Embedding image content to its url
    image_url = "data:image/png;base64," + figdata_png
    # Return json with image, target, prediction
    return jsonify({'image_url' : image_url, 'target': word, 'prediction' : prediction})
	
@app.route('/', methods=['GET'])
def index_page():
    return render_template('index.html')