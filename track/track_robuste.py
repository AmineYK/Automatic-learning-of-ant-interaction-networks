import cv2 as cv
import sys
import time
import pandas as pd
sys.path.append('.')

from fourmis import *

def get_random_colors(n):
    colors = []
    r = np.random.randint(0,255,n) / 255
    g = np.random.randint(0,255,n) / 255        
    b = np.random.randint(0,255,n) / 255
    
    return list(zip(r,g,b))


positions_nourriture = [((400,450),(170,170)) , ((370,60),(150,120)) ]
position_nid = ((600,800),(50,50))
position_couvain = ((210,850),(165,135))
position_reine = ((435,885),(50,50))


#v = cv.VideoCapture('video_boite_entiere-test.ts')
v = cv.VideoCapture('test_nid_vert.MTS')


bg_subtractor = cv.createBackgroundSubtractorMOG2()
history_length = 3000
bg_subtractor.setHistory(history_length)

erode_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (5, 7))

num_history_frames_populated = 0
cap = 0
gFourmis = GestionFourmis()
cpt = 0
dico_positions = {}
dico_nid = {}
dico_couvain = {}
dico_reine = {}
list_dico_nouriture = []

while True: # lire frame par frame
    ok, frame = v.read()
    frame = frame[:,200:1200]
    # Apply the MOG background subtractor.
    fg_mask = bg_subtractor.apply(frame)

    # Let the background subtractor build up a history.
    if num_history_frames_populated < history_length:
        num_history_frames_populated += 1
        continue
    colors = get_random_colors(len(positions_nourriture)) 

    #focus sur le nid
    (ni_x,ni_y),(ni_w,ni_h) = position_nid
    cv.putText(frame,"Nid",(ni_x,ni_y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
    cv.rectangle(frame,(ni_x,ni_y),(ni_x+ni_w,ni_y+ni_h),(142, 24, 42),2)
        
     #focus sur le couvain
    (couv_x,couv_y),(couv_w,couv_h) = position_couvain
    cv.putText(frame,"Couvain",(couv_x,couv_y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
    cv.rectangle(frame,(couv_x,couv_y),(couv_x+couv_w,couv_y+couv_h),(206, 225, 113),2)
        
    #focus sur la reine
    (rei_x,rei_y),(rei_w,rei_h) = position_reine
    cv.putText(frame,"Reine",(couv_x,couv_y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
    cv.rectangle(frame,(rei_x,rei_y),(rei_x+rei_w,rei_y+rei_h),(206, 225, 113),2)
    # Create the thresholded image.
    fg_mask = cv.GaussianBlur(fg_mask, (25,25),0) # enlever noise kernel (21,21)
    _, thresh = cv.threshold(fg_mask, 40, 255,
                                  cv.THRESH_BINARY)


    # focus sur les differentes zones de la nouriture
    for i_pos in range(len(positions_nourriture)):
        list_dico_nouriture.append({})
        (nou_x,nou_y),(nou_w,nou_h) = positions_nourriture[i_pos]
        cv.putText(frame,f"Nouriture {i_pos}",(nou_x,nou_y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        cv.rectangle(frame,(nou_x,nou_y),(nou_x+nou_w,nou_y+nou_h),(190, 100, 104),2)

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

        for i_pos in range(len(positions_nourriture)):
            dico_nouri = list_dico_nouriture[i_pos]
            (nou_x,nou_y),(nou_w,nou_h) = positions_nourriture[i_pos] 
            if x >= nou_x and x <= nou_x+nou_w and y >= nou_y and y <= nou_y+nou_h :
                if id in dico_nouri.keys():
                    (id,coun) = dico_nouri[id]
                    coun += 1
                    dico_nouri[id] = (id,coun)
                else:
                    dico_nouri[id] = (id,1)

        # nid
        (nou_x,nou_y),(nou_w,nou_h) = position_nid
        if x >= nou_x and x <= nou_x+nou_w and y >= nou_y and y <= nou_y+nou_h :
            if id in dico_nid.keys():
                id,coun = dico_nid[id]
                coun += 1
                dico_nid[id] = (id,coun)
            else:
                dico_nid[id] = (id,1)
                
        # couvain
        (nou_x,nou_y),(nou_w,nou_h) = position_couvain
        if x >= nou_x and x <= nou_x+nou_w and y >= nou_y and y <= nou_y+nou_h :
            if id in dico_couvain.keys():
                id,coun = dico_couvain[id]
                coun += 1
                dico_couvain[id] = (id,coun)
            else:
                dico_couvain[id] = (id,1)

        # reine
        (nou_x,nou_y),(nou_w,nou_h) = position_reine
        if x >= nou_x and x <= nou_x+nou_w and y >= nou_y and y <= nou_y+nou_h :
            if id in dico_reine.keys():
                id,coun = dico_reine[id]
                coun += 1
                dico_reine[id] = (id,coun)
            else:
                dico_reine[id] = (id,1)
        if id in dico_positions.keys():
            dico_positions[id].append([id,x,y,w,h])
        else:
            dico_positions[id] = [[id,x,y,w,h]]
        cv.putText(frame,f"fourmis {id}",(x,y),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv.circle(frame, (int(x + w/2),int(y + h/2)), radius=0, color=(0, 0, 255), thickness=-1)
    cv.imshow("frame",frame)
    #print(len(dico_positions))
    #print(dico_couvain)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    if cap == 35000:
        break
    cap += 1

v.release()
cv.destroyAllWindows()


def create_dataframe(dico,COLUMN_NAMES):
    nb_fourmis = len(list(dico.keys()))
    dataF = pd.DataFrame(columns=COLUMN_NAMES)
    for i in range(nb_fourmis):
        positions = np.array(dico.get(i))
        dataF_tmp = pd.DataFrame(positions,columns=COLUMN_NAMES)
        dataF = pd.concat([dataF,dataF_tmp],ignore_index=True)

    return dataF


COLUMN_NAMES = ['Id_ant','X','Y','Width','Height']
dataset_positions = create_dataframe(dico_positions,COLUMN_NAMES)
COLUMN_NAMES = ['Id_ant','temps_porte']

dataset_nid = pd.DataFrame(dico_nid.values(),columns=COLUMN_NAMES)

COLUMN_NAMES = ['Id_ant','temps_nouri0']
COLUMN_NAMES1 = ['Id_ant','temps_nouri1']

dataset_nouri0 = pd.DataFrame(list_dico_nouriture[0].values(),columns=COLUMN_NAMES)
dataset_nouri1 = pd.DataFrame(list_dico_nouriture[1].values(),columns=COLUMN_NAMES1)

COLUMN_NAMES = ['Id_ant','temps_couvain']

dataset_couvain = pd.DataFrame(dico_couvain.values(),columns=COLUMN_NAMES)

COLUMN_NAMES = ['Id_ant','temps_reine']

dataset_reine = pd.DataFrame(dico_reine.values(),columns=COLUMN_NAMES)
tmp1 = dataset_positions.merge(dataset_nid,on='Id_ant',how='outer').fillna(0)
tmp2 = dataset_nouri0.merge(dataset_nouri1,on='Id_ant',how='outer').fillna(0)
tmp3 = dataset_couvain.merge(dataset_reine,on='Id_ant',how='outer').fillna(0)

tmp1 = tmp1.merge(tmp2,on='Id_ant',how='outer').fillna(0)
dataset = tmp1.merge(tmp3,on='Id_ant',how='outer').fillna(0)

# supprimer l'id 0
dataset = dataset.drop(0)
dataset


dataset.to_csv('out.csv', index=False)
