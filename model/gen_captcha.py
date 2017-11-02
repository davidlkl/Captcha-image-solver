# Built-in Lib
import re
import os

# External Lib
from captcha.image import ImageCaptcha
from PIL import Image

# Global Variable
# Built-in English word dictionary in Ubuntu
DICTIONARY_FILE = '/usr/share/dict/words'
TRAIN_DIR = 'train'

# Clean up
def remove_output(file_dir):
    for file in os.listdir(file_dir):
        file_path = os.path.join(file_dir,file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Read system dictionary
# Words with length (4-6) are chosen to return a list of words
def read_dictionary():
    word_list = []
    for word in open(DICTIONARY_FILE):
        # Remove trailing end line char
        word = word.strip('\n')
        # Remove non-digit-or-english-letter char
        word = re.sub("[^a-zA-Z]","",word)
        if (len(word) > 3 and len(word)<7) and (not word in word_list):
            word_list.append(word)
    return word_list

def save_captcha_image(image):
    image.save(os.path.join(TRAIN_DIR, text + '.png'))
    return

def gen_captcha_image(imageCaptcha, text):
    captcha = imageCaptcha.generate(text)
    captcha_image = Image.open(captcha)
    return captcha_image

# Avoid creating ImageCaptcha instance repeatedly
def gen_captcha_images(words):
    imageCaptcha = ImageCaptcha(width=200, height=80)
    for word in words:
        captcha_image = gen_captcha_image(imageCaptcha, word)
        save_captcha_image(captcha_image)
    return

def main():
    remove_output(TRAIN_DIR)
    dictionary = read_dictionary()
    gen_captcha_images(dictionary)
