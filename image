from flask import Flask, render_template, request
import numpy as np
from sklearn.cluster import KMeans
import cv2
import datetime
import time

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_clusters = list(map(int, request.form['num_clusters'].split()))
        result_images = process_images(num_clusters)
        return render_template('index.html', result_images=result_images)
    return render_template('index.html', result_images=None)

def process_images(num_clusters_list):
    image1 = cv2.imread("mri1.jpg")
    image2 = cv2.imread("mri2.jpg")
    image3 = cv2.imread("Berlin.jpg")

    image = [image1, image2, image3]
    reshaped = [0, 0, 0]
    for i in range(0, 3):
        reshaped[i] = image[i].reshape(image[i].shape[0] * image[i].shape[1], image[i].shape[2])

    clustering = [0, 0, 0]
    for i in range(0, 3):
        kmeans = KMeans(n_clusters=num_clusters_list[i], n_init=40, max_iter=500).fit(reshaped[i])
        clustering[i] = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
                                    (image[i].shape[0], image[i].shape[1]))

    sorted_labels = [[], [], []]
    for i in range(0, 3):
        sorted_labels[i] = sorted([n for n in range(num_clusters_list[i])],
                                  key=lambda x: -np.sum(clustering[i] == x))

    kmeans_image = [0, 0, 0]
    concat_image = [[], [], []]
    for j in range(0, 3):
        kmeans_image[j] = np.zeros(image[j].shape[:2], dtype=np.uint8)
        for i, label in enumerate(sorted_labels[j]):
            kmeans_image[j][clustering[j] == label] = int((255) / (num_clusters_list[j] - 1)) * i
        concat_image[j] = np.concatenate((image[j], 193 * np.ones((image[j].shape[0], int(0.0625 * image[j].shape[1]), 3), dtype=np.uint8),
                                          cv2.cvtColor(kmeans_image[j], cv2.COLOR_GRAY2BGR)), axis=1)

    image_filenames = []
    for i in range(0, 3):
        dt = datetime.datetime.now()
        file_extension = "png"
        filename = (str(dt.hour)
                    + ':' + str(dt.minute) + ':' + str(dt.second)
                    + ' C_' + str(num_clusters_list[i]) + '.' + file_extension)
        cv2.imwrite(filename, concat_image[i])
        image_filenames.append(filename)
        time.sleep(1)

    return image_filenames

if __name__ == '__main__':
    app.run(debug=True)
