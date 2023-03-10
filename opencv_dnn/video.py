import numpy as np
import cv2

cap = cv2.VideoCapture('images\people.mp4')

if cap.isOpened() == False:
    print('Cannot open file or video stream')
    exit()

all_rows = open('models\synset_words.txt').read().strip().split('\n')
classes = [r[r.find(' ') + 1:] for r in all_rows]

net = cv2.dnn.readNetFromCaffe('models\\bvlc_googlenet.prototxt.txt', 'models\\bvlc_googlenet.caffemodel')

while True:
    ret, frame = cap.read()
    
    if ret == False:
        print('End of video file reached or read error occurred')
        break
    
    print('Read frame:', frame.shape)
    
    blob = cv2.dnn.blobFromImage(frame, 1, (224, 224))

    net.setInput(blob)

    outp = net.forward()
    
    r = 1
    for i in np.argsort(outp[0])[::-1][:5]:
        txt = ' "%s" probability  "%.3f" ' % (classes[i], outp[0][i] * 100)
        cv2.putText(frame, txt, (0, 25 + 40*r), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        r += 1
    
    cv2.imshow('Frame', frame)   
        
    if cv2.waitKey(25) & 0xFF == 27:
        break 
    
cap.release()
cv2.destroyAllWindows()
