import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps

x = np.load("image.npz")["arr_0"]
x = np.array(x)
y = pd.read_csv("labels.csv")["labels"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, train_size = 3500, test_size = 500)
x_train = x_train/255.0
x_test = x_test/255.0

clf = LogisticRegression(solver = "saga", multi_class = "multinomial")
clf.fit(x_train, y_train)

def getPrediction(img):
    im_pil = Image.open(img)
    im_bw = im_pil.convert("L")
    im_bw_resized = im_bw.resize((28, 28), Image.ANTIALIAS)

    pixel = 20
    min_pixel = np.percentile(im_bw_resized, pixel)
    im_bw_inverted = np.clip(im_bw_resized - min_pixel, 0, 255)

    max_pixel = np.max(im_bw_resized)
    im_bw_inverted = np.asarray(im_bw_inverted)/max_pixel

    sample = np.array(im_bw_inverted).reshape(1, 784)
    pred = clf.predict(sample)
    
    return pred[0]