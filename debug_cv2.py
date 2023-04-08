import tkinter
from tkinter import filedialog
import tkinter.font as tf
from PIL import Image


def upload_file():
    # askopenfilename 上传1个;askopenfilenames上传多个
    result_text.delete(1.0, 'end')
    black_pixel = 0
    white_pixel = 0
    select_file = tkinter.filedialog.askopenfilename()
    image = Image.open(select_file)

    width, height = image.size
    for i in range(0, width):
        for j in range(0, height):
            # 获取像素
            current_pixel = image.getpixel((i, j))
            # 白色
            if current_pixel == (255, 255, 255):
                white_pixel += 1
            # 黑色
            elif current_pixel == (0, 0, 0):
                black_pixel += 1
    black_pixel_percent = black_pixel / (width * height)
    white_pixel_percent = white_pixel / (width * height)

    result_text.insert("insert", f'黑色像素:{black_pixel},占比:{black_pixel_percent:.{2}%}'
                                 f'\n白色像素:{white_pixel},占比:{white_pixel_percent:.{2}%}')


root = tkinter.Tk()
root.title('黑白像素统计demo')
root.minsize(240, 160)
my_font = tf.Font(family='微软雅黑', size=15)  # 设置字体

main_frame = tkinter.Frame(root).grid()

# 选择文件按钮
choice_file_btn = tkinter.Button(main_frame, text='请选择文件', command=upload_file)
# 计算结果显示框
result_text = tkinter.Text(main_frame, width=30, height=5, font=my_font)

choice_file_btn.grid(row=0, column=0)
result_text.grid(row=0, column=1)

root.mainloop()
