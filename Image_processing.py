import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def crop_img(img):
    h,w = img.shape[0],img.shape[1]
    cropped_img = img[0:int(h/8),int((w-w/0.5)):int(w/2)]
    return cropped_img

def img_to_gray(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ret, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_TRUNC)
    return thresh