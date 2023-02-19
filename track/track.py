
'''
    © Copyright (C) 2023
    Collaborateurs : Amine YOUCEF KHODJA, Koceila KEMICHE, Hoang Son NGUYEN.*

'''


import cv2 as cv
import sys
import time
sys.path.append('.')

from fourmis import *

v = cv.VideoCapture('video_boite_entiere-test.ts')
bg = cv.imread("bg.png")
bg = cv.cvtColor(bg,cv.COLOR_BGR2GRAY)
bg = cv.GaussianBlur(bg, (21,21),0)


bg_subtractor = cv.createBackgroundSubtractorMOG2()
history_length = 250
bg_subtractor.setHistory(history_length)

erode_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (5, 7))

num_history = 0

gFourmis = GestionFourmis()
cpt = 0

while True: # lire frame par frame
    ok, frame = v.read()
    # Apply the KNN/MOG(Gauss Mix) background subtractor.
    fg_mask = bg_subtractor.apply(frame)

    # Historie pour background
    if num_history < history_length:
        num_history += 1
        continue
    # Thresh
    fg_mask = cv.GaussianBlur(fg_mask, (25,25),0) # enlever noise kernel (21,21)
    _, thresh = cv.threshold(fg_mask, 30, 255,
                                  cv.THRESH_BINARY)
    
    #cv.erode(thresh, erode_kernel, thresh, iterations=3)
    cv.dilate(thresh, dilate_kernel, thresh, iterations=3)    
    cnts, res = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # trouver contours dans thresh

    for c in cnts: # tracer box
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
        cv.putText(frame,f"fourmis {id}",(x,y),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv.imshow("frame",frame)
    cv.imshow("mask",fg_mask)
    #cv.imshow("thresh",thresh)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    # if i >500:
    #     break

v.release()
cv.destroyAllWindows()


 
