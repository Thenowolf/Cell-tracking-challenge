from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        # Místo POJO objektu využívám tříděný slovník, kdy ID odpovídá informaci ke konkrétnímu objektu
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.prevSpeed = OrderedDict()
        self.speedX = OrderedDict()
        self.speedY = OrderedDict()

        # Jen proměnná která uchovává po kolika snímcích má objekt zmizet z "trackování"
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.speedX[self.nextObjectID] = 0
        self.speedY[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            # Označ jako ztracené
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # Pokud je ztracený víc než X framu, smaž...
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        # Inicializace proměnné
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # Z čtyřuhelníku získej centroid - prostředek
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # Pokud nic netrackuju, registruj
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # Pokud trackuju, snažím se přiřazit trackovaný objekt k objektu z nového snímku
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            #Vypočti vzdálenost
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for i,(row, col) in enumerate(zip(rows, cols)):

                if row in usedRows or col in usedCols:
                    continue

                # Výpočty v kombinaci s předchozím framem a nynějším + nějaké přepsání hodnot
                objectID = objectIDs[row]
                self.prevSpeed[objectID] = self.objects[objectID]
                self.objects[objectID] = inputCentroids[col]
                if self.prevSpeed[objectID][0] or self.prevSpeed[objectID][1] is not None:
                    test = self.prevSpeed[objectID][0]
                    test2 = inputCentroids[col][0]
                    self.speedX[objectID] += abs(inputCentroids[col][0] - self.prevSpeed[objectID][0])
                    self.speedY[objectID] += abs(inputCentroids[col][1] - self.prevSpeed[objectID][1])
                self.disappeared[objectID] = 0

                # Vedu si záznam jaký sloupec a řádek jsem už "zpracoval"
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # Dívám se které hodnoty jsem nevyužil a kdyžtak označím jako zmizelé případně smažu
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # Případně registruju jako nový objekt
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects, self.prevSpeed, self.speedX, self.speedY