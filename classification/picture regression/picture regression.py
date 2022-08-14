import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.cluster import KMeans
import numpy as np

img = imread(r'C:\Users\ed069\Desktop\diabetes\IMG_20191009_213658_648.jpg')
img_size = img.shape
X = img.reshape(img_size[0]*img_size[1],img_size[2])
for i in range(30):
    km = KMeans(n_clusters= i+1)
    km.fit(X)
    X_compressed = km.cluster_centers_[km.labels_]
    X_compressed = np.clip(X_compressed.astype('uint8'),0,255)
    X_compressed = X_compressed.reshape(img_size[0], img_size[1], img_size[2])
    fig, ax =plt.subplots(1,2,figsize = (12,8))
    ax[0].imshow(img)
    ax[0].set_title('original Image')
    ax[1].imshow(X_compressed)
    ax[1].set_title('compressed Image with {i+1} colors')
    for ax in fig.axes:
      ax.axis('off')
    plt.tight_layout()