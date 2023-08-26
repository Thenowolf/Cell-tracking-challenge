import numpy as np
import cv2
import time
import centroidtracker


def celltracking(video_path):
    cap = cv2.VideoCapture(video_path)
    #Inicializace centroid trackeru
    cd = centroidtracker.CentroidTracker(maxDisappeared=3)
    loopnum = 0
    #Sekundární sekvence - "trails"
    sec_frame = np.zeros((904, 1224,3), dtype = "uint8")
    while True:
        ret, frame = cap.read()
        #Převod na formát, ve kterém mužu hledat
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Hledám kontury v obrázku
        p0, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        #Spočítám si čtyřuhelník z označených objektu
        for i in range(len(p0)):
            x,y,w,h = cv2.boundingRect(p0[i])
            rect = [x, y, x+w, y+h]
            rects.append(rect)
            #Vykreslím čtyřúhelník
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        # Přidám všechny "objekty" do trackeru, objekty jsou čtyřuhelníky
        objects, prev, speedX, speedY = cd.update(rects)

        #Vykresluji informace co potřebuji + malé dopočty
        for i, (objectID, centroid) in enumerate(objects.items()):
            text = "ID {}".format(objectID)
            SpeedX = "AVG.SpeedX {}".format(round(speedX[objectID]/(loopnum+1),2))
            SpeedY = "AVG.SpeedY {}".format(round(speedY[objectID]/(loopnum+1),2))
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
            cv2.putText(frame, SpeedX, (centroid[0] - 7, centroid[1] - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.20, (255, 0, 255), 1)
            cv2.putText(frame, SpeedY, (centroid[0] - 5, centroid[1] - 5),
                cv2.FONT_HERSHEY_DUPLEX , 0.20, (255, 0, 255), 1)
            cv2.circle(frame, (centroid[0], centroid[1]), 1, (0, 255, 0), -1)
            if loopnum > 0:
                try:
                    cv2.arrowedLine(sec_frame, (prev[objectID][0], prev[objectID][1]), (centroid[0], centroid[1]), (0, 0, 255), 1)
                    cv2.arrowedLine(frame, (prev[objectID][0], prev[objectID][1]), (centroid[0], centroid[1]), (0, 0, 255), 2)
                except:
                    print("Nenalezl jsem puvodni zdroj")


        cv2.imshow("Trail", sec_frame)
        cv2.imshow("Countours", frame)
        k = cv2.waitKey(500) & 0xFF
        loopnum += 1

celltracking("/Users/denislokaj/Documents/PROJEKTBAAD/cells/unet-pred-binarize/A2_03_2_1_DAPI_%03d.png")
