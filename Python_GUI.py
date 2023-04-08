from http.client import IncompleteRead
import tkinter
from tkinter import filedialog
import tkinter.font as tf
import cv2
import pytesseract
from PIL import Image


def upload_file():
    result_text.delete(1.0, 'end')
    select_file = tkinter.filedialog.askopenfilename()
    image = Image.open(select_file)
    pytesseract.pytesseract.tesseract_cmd = "D://Tesseract-Ocr5.0//tesseract.exe"
    text = pytesseract.image_to_string(select_file)
    
    result_text.insert("insert", f'Identify Results:{text}')
    
    

if __name__ == '__main__':
    root = tkinter.Tk()
    root.title('Answer Sheet')
    root.minsize(600, 400)
    my_font = tf.Font(family='microsoft yahei', size=15)  # Set font

    main_frame = tkinter.Frame(root).grid()


    # Choose picture button
    choice_file_btn = tkinter.Button(main_frame, text='Please choose image', command=upload_file)
    # Recognize result show area
    result_text = tkinter.Text(main_frame, width=35, height=20, font=my_font)

    choice_file_btn.grid(row=0, column=0)
    result_text.grid(row=0, column=1)

    root.mainloop()
