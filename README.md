# Solver for English word captcha images

It is a program for training a model that can recognize english words (4-6 character lengths). Using a trained model, a simple web API is developed to return a random English word captcha image, target English word and prediction from the model.

##### Environment
Ubuntu 16.04, Python 3.5.2

##### Three main sections
1. Data preparation
2. Model Training
3. Local Server Setup

Flowchart:
![flowchart](/model/flowchart.jpg)

### Data Preparation
A library named Captcha is used for generating captcha from given text. <br>
Source: https://github.com/lepture/captcha

##### Training Dataset:
Words with 4-6 character length are selected from the built-in english word dictionary ("/usr/share/dict/word") in Ubuntu. For each word, a captcha image in png format with height 80, width 200 and channel 3 is generated. In total, 20684 captcha images are generated and stored in folder("model/train").

##### Relevant Files:
The code is saved as gen_captcha.py in folder("model").

### Model Training
Keras + Tensorflow-gpu are used for training the model.<br>
The chaptcha image dataset is split into 3 parts (train, validation, testing) in the ratio of 0.68: 0.17: 0.15.

##### Model Architecture:
I started with the one mentioned in this article (http://www.jianshu.com/p/25655870b458) and made some changes. <br>
![Model_Arc](/model/archi.jpg)

After 300 epochs of training, the model with lowest validation loss (0.12) is chosen. It has ~ 98% accuracy in identifying individual characters, roughly 0.98^5=90.4% accuracy in identifying the whole captcha image.<br>

##### Relevant Files:
The code is saved as train_model.py in folder("model"). The trained model is stored in .hdf5 format in folder ("static").

### Local Server Setup
A common python web framework Flask is used.
Source: http://flask.pocoo.org/

##### Relevant Files:
The relevant files include my_server.py in root folder and index.html in folder("templates").

##### Description:
The front-end is a simple webpage with a single button to generate random captcha image and the prediction made by the trained model. With the use of JQuery, the button will fire an ajax call to the server. The server will handle this request and return a json containing the captcha image and predition.

### Setting up the local server
##### Install the libraries:
`pip3 install -r requirements.txt`<br>


##### Set environment variable FLASK_APP:
`export FLASK_APP=my_server.py`

##### Run local server:
`flask run`
A local server will be run on localhost:5000.

### Heroku - a free web hosting website
### *** Not working now (For reference only) ***
This repository is pushed to Heroku, a free web hosting website. <br>
Link: https://davidlkl-captcha-solver.herokuapp.com/<br>
Please note that the server may be down for serveral hours per day becuase of the free plan and it takes some time for the server to reboot from sleep mode.

##### Relevant files:
Some configuaration files are needed to setup Heroku. They are Procfile, nltk.txt, requirements.txt and runtime.txt.

