import numpy as np
import argparse
import cv2
from imutils.video import VideoStream
import time
import imutils
import Person

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
PROTOTEXT = 'MobileNetSSD_deploy.prototxt'
CAFFEMODEL = 'MobileNetSSD_deploy.caffemodel'

persons = []
totalDown = 0
totalUp = 0
# load our serialized model from disk
# print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTEXT, CAFFEMODEL)
abac = "rtsp://admin:islabac123@168.120.33.119"
# print("[INFO] starting video stream...")
vs = VideoStream(src="lib2.mp4").start()
time.sleep(2.0)
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    # frame_copy = frame.copy()

    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                 (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    # print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    x1 = 0
    x2 = W
    y1 = int(H // 2 - (0.05 * H))
    y2 = int(H // 2 + (0.15 * H))
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            # print(CLASSES[idx])
            if CLASSES[idx] == "person":

                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)
                new = True
                for i in persons:
                    if abs(cX - i.getX()) <= W and abs(cY - i.getY()) <= H:
                        new = False
                        i.updateCoords(cX, cY)
                        if i.UP(y1, y2) == True:
                            totalUp += 1
                            write(totalUp,totalDown)
                            i.done = True
                        elif i.DOWN(y1, y2) == True:
                            totalDown += 1
                            write(totalUp,totalDown)
                            i.done = True
                    if i.timedOut():
                        index = persons.index(i)
                        persons.pop(index)
                        del i
                if new == True:
                    p = Person.MyPerson(cX, cY)
                    persons.append(p)

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


    # bottom line
    cv2.line(frame, (x1, y1), (x2, y1), (255, 0, 0), 2)
    # top line
    cv2.line(frame, (x1, y2), (x2, y2), (0, 0, 255), 2)

    info = [
        ("Out", totalUp),
        ("In", totalDown),
    ]
    def write(up, down):
        f = open("count.txt", "w")
        Up = str(up)
        Down = str(down)
        f.write("{},{}\n".format(Up,Down))
        f.close()
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
print(totalDown)
print(totalUp)
cv2.destroyAllWindows()
vs.stop()