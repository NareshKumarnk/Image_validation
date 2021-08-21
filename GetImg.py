import cv2
import numpy as np
from flask import Flask, request, jsonify
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

app = Flask(__name__)

def process_image(file):
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img

def mse(imageA, imageB):
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    return s

def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]

    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img

@app.route("/imagevalidate", methods=['POST'])
def two_img():
    file1 = request.files['image1'].read()
    file2 = request.files['image2'].read()
    original = process_image(file=file1)
    story = process_image(file=file2)

    h1, w1, c1 = original.shape
    h2, w2, c2 = story.shape
    h = min(h1, h2)
    w = min(w1, w2)

    original_center = center_crop(original, (w, h))
    story_center = center_crop(story, (w, h))

    original_center = cv2.cvtColor(original_center, cv2.COLOR_BGR2GRAY)
    story_center = cv2.cvtColor(story_center, cv2.COLOR_BGR2GRAY)

    index = compare_images(original_center, story_center)
    return jsonify({'index': index, 'h': h, 'w': w, 'shape': [original_center.shape[0], original_center.shape[1]]})

if __name__ == "__main__":
    app.run(debug=False,port=6000)
