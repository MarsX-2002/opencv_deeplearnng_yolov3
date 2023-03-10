import numpy as np
import cv2

# Load the image
img = cv2.imread('images\dog.jpg')


all_rows = open('models\synset_words.txt.txt').read().strip().split('\n')

classes = [r[r.find(' ') + 1:] for r in all_rows]

net = cv2.dnn.readNetFromCaffe('models\\bvlc_googlenet.prototxt.txt', 'models\\bvlc_googlenet.caffemodel')

blob = cv2.dnn.blobFromImage(img, 1, (224, 224))

net.setInput(blob)

outp = net.forward()
# print(outp)

# top 5 predictions
idx = np.argsort(outp[0])[::-1][:5]

for (i, id) in enumerate(idx):
    print('{}. {} ({}): Probability {:.3}%'.format(i + 1, classes[id], id, outp[0][id] * 100))

# for (i, c) in enumerate(classes):
#     if i == 4:
#         break
#     print(i, c)
    
# Display the smaller image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

