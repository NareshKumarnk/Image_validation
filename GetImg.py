import cv2
import numpy as np
from flask import Flask, request, jsonify
from skimage.metrics import structural_similarity as ssim
from Image_processing import *

app = Flask(__name__)

def process_image(file):
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img

def compare_images(image_1, image_2):
    s = ssim(image_1, image_2)
    return s

def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]

    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]

    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    return crop_img

def validated(text, image1, image2):
    for val in text.split('\n'):
        if ('my' in val.lower()) or ('me' in val.lower()) and ('yesterday' not in val.lower()):
            index = compare_images(image1, image2)
            break
        else:
            index = 0
    return index

@app.route("/imagevalidate", methods=['POST'])
def two_img():
    file1 = request.files['image1'].read()
    file2 = request.files['image2'].read()
    original = process_image(file=file1)
    story = process_image(file=file2)

    cv2.imshow("Image", crop_img(story))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    text = image_to_text(story)
    print(text)
    h1, w1, c1 = original.shape
    h2, w2, c2 = story.shape
    h = min(h1, h2)
    w = min(w1, w2)

    original_center = center_crop(original, (w, h))
    story_center = center_crop(story, (w, h))

    score = validated(text=text, image1=original_center, image2=story_center)

    if score >= 0.80:
        result = 'Similar'
    else:
        result = 'Not similar'

    return jsonify({'index': score, 'message': result, 'shape': [original_center.shape[0], original_center.shape[1]]})

if __name__ == "__main__":
    app.run(debug=False, port=6000)
