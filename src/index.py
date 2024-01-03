import os
import joblib
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import tkinter as tk

# 假设图片统一缩放到100x100像素


def load_images_from_folders(folder_path, label):
    images = []
    labels = []

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path).convert('RGB')

        # 将图片缩放到指定大小
        desired_size = (100, 100)
        img_resized = img.resize(desired_size, Image.Resampling.LANCZOS)

        # 转换为numpy数组并归一化到[0,1]
        img_nd_arr = np.array(img_resized) / 255.0
        images.append(img_nd_arr.flatten())
        labels.append(label)

    images_nd_arr = np.array(images)
    labels_nd_arr = np.array(labels)

    print('shape', images_nd_arr.shape)

    return images_nd_arr, labels_nd_arr


train_cat_folder = 'data/train/cat'
train_dog_folder = 'data/train/dog'

x_train_cat, y_train_cat = load_images_from_folders(train_cat_folder, 0)
x_train_dog, y_train_dog = load_images_from_folders(train_dog_folder, 1)

# 合并数据集
x_train = np.concatenate([x_train_cat, x_train_dog], axis=0)
y_train = np.concatenate([y_train_cat, y_train_dog])

# 对于逻辑回归，标签需要被编码为二元形式
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# 分割训练集和验证集（或测试集）
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train_encoded, test_size=0.2, random_state=42)

# 创建逻辑回归模型
# 求解器设置为saga，对大规模的稀疏数据又更好的处理
# 这里出现了一个问题时，一直没有收敛，反复振荡，推断可能是有一些干扰的特征，我们加大一下正则化惩罚力度C
estimator = LogisticRegression(max_iter=30, solver='saga', C=0.1)

# 训练模型
estimator.fit(x_train, y_train)
joblib.dump(estimator, 'model/cat-dog.pkl')
# 预测（这里假设我们已经有了测试集）
# x_test_cat, y_test_cat 和 x_test_dog, y_test_dog 应该按照类似的方式加载和合并
# ...

# 验证模型性能
accuracy = estimator.score(x_test, y_test)
# 试了下，逻辑回归在这个场景下，似乎只能达到 59.5%
print("准确率:", accuracy)

# 可视化校验
window = tk.Tk()
window.title("Cat-Dog")
