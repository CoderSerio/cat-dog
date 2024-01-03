import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np


class CatDogClassifierGUI:
    def __init__(self, model):
        self.root = tk.Tk()
        self.root.title("Cat-Dog")
        self.model = model
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # 上传按钮
        self.upload_button = tk.Button(
            self.root, text="UPLOAD IMAGE",
            command=self.load_image
        )

        self.upload_button.pack()

        # 结果标签
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

        self.root.mainloop()

    def load_image(self):
        file_path = tk.filedialog.askopenfilename(
            filetypes=[('Image Files', '*.jpg;*.png')])

        # 读取图片
        img = Image.open(file_path).convert('RGB')

        # 先在窗口展示图片
        window_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=window_img)
        self.image_label.image = window_img  # 防止图片被垃圾回收

        # 归一化
        desired_size = (100, 100)
        img_resized = img.resize(desired_size, Image.Resampling.LANCZOS)
        img_nd_arr = np.array(img_resized) / 255.0

        images = []
        images.append(img_nd_arr.flatten())
        images_nd_arr = np.array(images)

        print('shape', img_nd_arr.shape)

        prediction = self.model.predict(images_nd_arr)

        res = prediction[0]
        if res == 0:
            self.result_label.config(text="这是猫猫")
        else:
            self.result_label.config(text="这是狗狗")


if __name__ == "__main__":
    # 假设已经训练好了一个模型并保存了
    model = joblib.load('model/cat-dog.pkl')

    gui = CatDogClassifierGUI(model)
