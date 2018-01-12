from skimage import io
from skimage import transform
import numpy as np
import sys, os


data = []

all_images = os.path.join(sys.argv[1], '*.jpg')
images = io.imread_collection(all_images)
tmp = sys.argv[2].split('.')
target = int(tmp[0])

for image in images:
    new_img = transform.resize(image, (300 , 300, 3))
    new_img = np.array(new_img)
    new_img = new_img.flatten()
    data.append(new_img)

data = np.array(data)
mean = np.mean(data, axis = 0)

for i in range(len(data)):
    data[i] = data[i] - mean

trans_data = np.transpose(data)


U, S, V = np.linalg.svd(trans_data, full_matrices=False)

weights = []
for i in range(4):
    weights.append(np.dot(data[target], U[:, i]))
   
M = [0] * 270000
M = np.array(M)
for i in range(4):
    M = M + weights[i] * np.transpose(U[:, i])
M = M + mean

M -= np.min(M)
M /= np.max(M)
M = (M * 255).astype(np.uint8)

io.imsave('restruction.jpg', M.reshape(300,300,3))

