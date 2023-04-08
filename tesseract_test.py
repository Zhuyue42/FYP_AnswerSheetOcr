import pytesseract
from PIL import Image

image = Image.open('D://Year4//FYP//Project//305.jpg')
pytesseract.pytesseract.tesseract_cmd = "D://Tesseract-Ocr5.0//tesseract.exe"
text = pytesseract.image_to_string('D://Year4//FYP//Project//305.jpg')
print(text)
image.show()