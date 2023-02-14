'''
    © Copyright (C) 2023
    Collaborateurs : Amine YOUCEF KHODJA, Koceila KEMICHE, Hoang Son NGUYEN.*

'''


import cv2
import numpy as np


////////////////////////////////////////////////////////////
/////////////////////'''CLASSES '''/////////////////////////
////////////////////////////////////////////////////////////


'''
    Redefinition de la classe Kelman Filtre
'''
class KalmanFilter:
    
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0,1, 0], [0, 1, 0,1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        
        
    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y


////////////////////////////////////////////////////////////////
/////////////////////''' FONCTIONS '''/////////////////////////
///////////////////////////////////////////////////////////////

''' 
- Positions = pos1 , pos2 , .... , posn
- Path = pos1 -> pos2 -> pos3 -> .. -> posn
- le chemin  :  pos1 -> pos2 -> pos3 -> .. -> posn -> ? ? ? ? ? ?  -> position_de _reprise -> ... -> position_arret
- le but est d'estimer par le filtre de Kalman les positions manquantes sur le chemin de la fourmi ( ??? ) 
'''
def burn_in(kf,positions):
    
    '''Entrainer le model du point initial (0,0) jusqu'à l'arrivée à la position "posn" '''
    '''Normalement s'il y'a assez de points parcourus le model va reussir à converger vers une position tres
     proche de "posn"
     sinon il faut creer une sequence fictive de points assez grande pour faire converger le model vers "posn"
     Tous ca , sera dans le param positions autrement dit avant l'appelle de la fonction burnin
    '''
    # c'est la periode de "Burn-in"
    return [kf.predict(pos[0],pos[1]) for pos in positions]
    

def go_to_convergence(kf,path,position_reprise,niter_max=1000000,epsilon=100,verbose=False):
    # position de reprise 
    x_reprise , y_reprise = position_reprise
    positions = []
    n_iter = 1
    # recuperer toutes les positions parcourues par le model pour atteindre la "posn"
    burnin = burn_in(kf,path)
    # recuperer la "posn" predite par le model 
    pred_position_x,pred_position_y = burnin[len(burnin)-1]
    while n_iter < niter_max:
        # predire la position qui suit "posn" autrement dit la premiere position manquante
        predicted = kf.predict(pred_position_x,pred_position_y)
        # test de convergence
        # si la position perdite est assez proche de la position de reprise ( < à l'hyperparamtere epsilon)
        # alors convergence
        if np.abs(predicted[0]-x_reprise) < epsilon and np.abs(predicted[1]-y_reprise) < epsilon :
            positions.append(predicted)
            break
        
        # sauvgrade de la suite de positions
        positions.append(predicted)
        # mettre à jour la position precedente pour la prochaine prediction
        pred_position_x , pred_position_y = predicted
        n_iter += 1
        
    if verbose:
        print("Convergé au bout de ",n_iter," positions")
        
    # retourne les positions manquantes à predire , les positions du burn-in , nombre de positions atteinte à convergence
    return positions,burnin,n_iter


def show_path_position(positions,bg):
    img = cv2.imread(bg1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    for pos in position:
        cv2.circle(img,pos,15,(0,20,220),-1)

    cv2.namedWindow(bg+"_test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(bg+"_test", 1500, 1000)    
    cv2.imshow(bg+"_test",img)
    cv2.waitKey(0)


//////////////////////////////////////////////////////////
/////////////////////''' TEST '''/////////////////////////
//////////////////////////////////////////////////////////


# Definition du filtre
kf = KalmanFilter()
# definition de la position de reprise
reprise_pos = (1700, 690)
# positions = [(50, 100), (120, 241), (150, 300), (200, 330), (250, 380), (200, 380), (250, 400)]
positions = [(4, 300), (61, 256), (116, 214), (170, 180), (225, 148), (279, 120), (332, 97),
         (383, 80), (434, 66), (484, 55), (535, 49), (586, 49), (634, 50),
         (683, 58), (731, 69), (778, 82), (824, 101), (870, 124), (917, 148),
         (962, 169), (1006, 212), (1051, 249), (1093, 290)]
pred_positions , burnin , nb_positions = go_to_convergence(kf,positions,final_pos,verbose=True)

bg = 'bg.jpg'
# affichage des positions predites     
show_path_position(pred_positions,bg)

# affichage des positions parcourues pendant le burnin   
show_path_position(burnin,bg)