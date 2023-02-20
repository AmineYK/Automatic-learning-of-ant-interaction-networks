
'''
    © Copyright (C) 2023
    Collaborateurs : Amine YOUCEF KHODJA, Koceila KEMICHE, Hoang Son NGUYEN.*

'''


import numpy as np
import cv2 as cv
from fourmis import *
from sys import argv


## argv[1]: longeur de trace, -1: all, 0: none, n : dernier n frames

gFourmis = GestionFourmis()
cpt = 0


video = cv.VideoCapture('video_boite_entiere-test.ts')

ret, frame = video.read()
height, width, layers = frame.shape


fgbg = cv.createBackgroundSubtractorMOG2()

noy_len = 8
noyau = np.ones((noy_len,noy_len))/(noy_len**2)
ddepth = -1

def blur(image):
    return cv.filter2D(image,ddepth,noyau)

thresh_val = 100

while True:
    _, frame = video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame)
    masque = blur(fgmask)
    _, masque = cv.threshold(masque, thresh_val, 255, cv.THRESH_BINARY)
    
    cons, _ = cv.findContours(masque, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    img = (1 - masque)*gray
    for c in cons: # tracer box
        id = None
        (x,y,w,h) = cv.boundingRect(c) # coord
        if gFourmis.getNbFourmis() == 0:
            fourmis = Fourmis(cpt, x,y,w,h)
            gFourmis.add(fourmis)
            id = fourmis.getId()
            cpt +=1
        else:
            f_tmp = Fourmis(cpt, x,y,w,h)
            centre = np.array([x + w/2,y + h/2])
            d = gFourmis.getListe()
            id_chercher = trouvePlusProche(f_tmp, d)
            if id_chercher == -1:
                gFourmis.add(f_tmp)
                id = f_tmp.getId()
                cpt +=1
            else:
                gFourmis.majFourmis(id_chercher, (centre,None))
                id =id_chercher
        cv.putText(frame,f"{id}",(x,y),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        if int(argv[1]) != 0:
            len_trace = int(argv[1])
            for i in range(cpt):
                cv.polylines(frame, np.int32([gFourmis.getPath(i, len_trace)]), False, (200, 0, 100), thickness=1)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()


 
