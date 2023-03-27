import os
import cv2
from PIL import Image
from ultralytics import YOLO

class yolov8:
    def __init__(self, model = "yolov8m.pt", filter = [], conf= 0.75):
        self.model = YOLO(model)
        self.filter= filter
        self.conf = conf
    def pred(self,img):
        results = self.model.predict(source=img)
        for result in results: #assume 1 image input
            boxes = result.boxes.cpu().numpy()

        finalboxes=[]
        for box in boxes:
            if(box.conf > self.conf):
                if(len(self.filter)>0):
                    for f in self.filter:
                        if(box.cls==f):
                            finalboxes.append(box)
                else:
                    finalboxes.append(box)
        return finalboxes




    def crop(self, img, box):
        box = box.xyxy[0]
        crop = img[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
        return crop
    def cropall(self, img, boxes):
        crops = []
        for box in boxes:
            crops.append(self.crop(img, box))
        return crops


if __name__ == "__main__":
    img = cv2.imread("testimgs/a.jpg")
    img = cv2.imread("testimgs/b.png")
    img = cv2.imread("testimgs/c.png")
    img = cv2.imread("testimgs/e.png")
    yolo = yolov8()
    boxes = yolo.pred(img)
    crops = yolo.cropall(img,boxes)



    for crop in crops:

        cv2.imshow("", crop)
        cv2.waitKey(0)
