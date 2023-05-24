import cv2 as cv
import sys
import copy as cp
sys.path.append('.')

from fourmis import *

v = cv.VideoCapture('video_boite_entiere-test.ts')
#v = cv.VideoCapture('test_nid_vert.MTS')

bg = cv.imread("bg.png")
bg = cv.cvtColor(bg,cv.COLOR_BGR2GRAY)
bg = cv.GaussianBlur(bg, (21,21),0)

cpt = 0
gFourmis = GestionFourmis()
i = 0
while True: # lire frame par frame
    ok, frame = v.read()
    thresh = None
    if i:
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) # couleur -> gray
        gray = cv.GaussianBlur(gray, (21,21),0) # enlever noise kernel (21,21)

        diff = cv.absdiff(bg,gray) # comparer avec le background
        thresh = cv.threshold(diff,35,255,cv.THRESH_BINARY)[1] # si sup a 30 -> 255, sinon 0
        thresh = cv.dilate(thresh,None,iterations = 2) # rendre plus epaise
        cnts, res = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # trouver contours dans thresh
        f_sans_boite = cp.deepcopy(frame)
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
            cv.imshow("thresh",thresh)
        #if i == 720:
            #cv.imwrite(f"dectection_sans_boite.png", f_sans_boite[:,300:1200])
            #cv.imwrite(f"detection_thresh.png",thresh[:,300:1200])
            #cv.imwrite(f"dectection_avec_boite.png", frame[:,300:1200])
            #print("fini")
            #break
    #cv.imshow("frame",frame)
    #cv.imshow("thresh",thresh)
    print(i)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    if i == 200:
        cv.imwrite(f"detection_avant.png", frame[:,300:1200])
    if i == 500:
        cv.imwrite(f"detection_apres.png", frame[:,300:1200])
        break
    i+=1
    # if i >500:
    #     break

v.release()
cv.destroyAllWindows()
