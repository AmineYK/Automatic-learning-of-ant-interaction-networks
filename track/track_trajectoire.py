import numpy as np
import cv2 as cv
from fourmis import *
from sys import argv
import time

## argv[1]: longeur de trace, -1: all, 0: none, n : dernier n frames

gFourmis = GestionFourmis()
cpt = 0


video = cv.VideoCapture('video_boite_entiere-test.ts')
#video = cv.VideoCapture('test_nid_vert.MTS')

ret, frame = video.read()
height, width, layers = frame.shape


fgbg = cv.createBackgroundSubtractorMOG2()

noy_len = 8
noyau = np.ones((noy_len,noy_len))/(noy_len**2)
ddepth = -1

def blur(image):
    return cv.filter2D(image,ddepth,noyau)
thresh_val = 100

def mouseClick(events, x, y, flags, param):
    # Pour trouver la position en cliquant sur video
    if events == cv.EVENT_LBUTTONDOWN:
        print(x, y)

def estDansNid(centre, nid):
    return cv.pointPolygonTest(nid, centre, False) < 0

x_nourri = 750
w_nourri = 100
y_nourri = 450
h_nourri = 130
nourri_zone = (x_nourri, y_nourri, w_nourri, h_nourri)

nid_top_left = [539,801]
nid_top_right = [1060,834]
nid_bot_right = [1048,1066]
nid_bot_left = [532,1043]
nid = np.array([nid_top_left,nid_bot_left,nid_bot_right,nid_top_right,nid_top_left])

cpt_nourri = 0
cpt_commu = 0
cpt_commu_nid = 0

def estDedans(zone, objet):
    x, y, w, h = zone
    x_obj, y_obj = objet
    if x_obj >= x and x_obj <= x+w and y_obj >= y and y_obj <= y+h:
        return True
    return False

def verifOverlap(l1, r1, l2, r2, seuil = None):
    # rectangle surface 0
    if l1[0] == r1[0] or l1[1] == r1[1] or r2[0] == l2[0] or l2[1] == r2[1]:
        return False
     
    # separe horizontalement
    if l1[0] > r2[0] or l2[0] > r1[0]:
        return False
 
    # separer verticalement
    if r1[1] > l2[1] or r2[1] > l1[1]:
        return False
    return True

def veriOverlap_list(l1, r1, liste_rec, seuil = None):
    for rec in list_rec:
        if verifOverlap(l1, r1, rec[0], rec[1]):
            return True
    return False

def aMerger(p, list_cons):
    cpt = 0
    for c in list_cons:
        (x,y,w,h) = cv.boundingRect(c) # coord
        centre = np.array([x + w/2,y + h/2])
        if np.sqrt((p[0]-centre[0])**2 + (p[1]-centre[1])**2) <5:
            cpt +=1
            if cpt >= 2:
                return True
    return False

con_pred = None
j=0
while True:
    #time.sleep(0.1)
    _, frame = video.read()
    if j< 2000:
        j+=1
        continue
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame)
    #masque = cv.medianBlur(fgmask, 5)
    masque = blur(fgmask)
    _, masque = cv.threshold(masque, thresh_val, 255, cv.THRESH_BINARY)
    
    cons, _ = cv.findContours(masque, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    img = (1 - masque)*gray

    for c in cons: # tracer box
        id = None
        (x,y,w,h) = cv.boundingRect(c) # coord
        centre = np.array([x + w/2,y + h/2])
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
            if estDedans(nourri_zone, centre):
                path = gFourmis.getPath(id_chercher, 0)
                if len(path) >=2 and not estDedans(nourri_zone, path[-2]):
                    cpt_nourri += 1
        #cv.putText(frame,f"{id}-{w*h}-{not estDansNid(centre, nid)}",(x,y),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        if con_pred is not None and (w*h) >= 600 and not estDansNid(centre, nid) and aMerger(centre, con_pred): # exterieur
            cpt_commu += 1
        if (w*h) > 400 and estDansNid(centre, nid):
            if id:
                print(f"ici {id} {w*h} {estDansNid(centre, nid)} {centre}")
            cpt_commu_nid += 1
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

        if int(argv[1]) != 0:
            len_trace = int(argv[1])
            for i in range(cpt):
                cv.polylines(frame, np.int32([gFourmis.getPath(i, len_trace)]), False, (200, 0, 100), thickness=1)
    con_pred = cons
    #cv.rectangle(frame,(x_nourri,y_nourri),(x_nourri+w_nourri,y_nourri+h_nourri),(100,60,100),3)
    #cv.putText(frame,f"Nb de visites {cpt_nourri}",(50,370),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    #cv.putText(frame,f"NbCommu ext {cpt_commu}",(50,600),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    #cv.putText(frame,f"NbCommu nid {cpt_commu_nid}",(50,800),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    cv.imshow('frame', frame)
    if j == 2700:
        cv.imwrite(f"trajectoire.png",frame[:,300:1200])
        break
    j+=1
    #cv.imshow('mask',masque)
    #cv.setMouseCallback('frame', mouseClick)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()


