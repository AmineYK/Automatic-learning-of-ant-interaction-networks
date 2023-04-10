
import numpy as np 
import matplotlib.pyplot as plt
import cv2

    
##################################################  
################ DATA PREPARATION ################  
##################################################  

def discretisation_uniforme_grille(grid_h,grid_w,bins_h,bins_w):
    """discretisation de la grille en intervalles 2D
    A partir de la discretisation extraire les coordonnées des espaces sous 4 points
    par exemple une grille de (100,500) discretiser en (2,2) on aura : les pas de discretisations respectifs 50 et 250 
    --> [0,50,100] et [0,250,500] ainsi on aura 2*2 espaces disjoints """
    inter_h = [0]
    inter_w = [0]
    res = []
    pas_h = int(grid_h / bins_h)
    pas_w = int(grid_w / bins_w)
    
    avant = 0
    for i in range(bins_h):
        inter_h.append(avant + pas_h)
        avant = avant + pas_h
    
    avant = 0
    for i in range(bins_w):
        inter_w.append(avant + pas_w)
        avant = avant + pas_w

    for i in range(bins_h):  
        for j in range(bins_w):  
            res.append((inter_h[i],inter_w[j]))
            
    formes = []
    for couple in res:
        x,y = couple
        formes.append([(x,y),(x,y+pas_w),(x+pas_h,y),(x+pas_h,y+pas_w)])
        
    return formes

def indi_position_in_forme(formes,position):
    x , y = position
    for i_forme in range(len(formes)):
        NW,SW,NE,SE = formes[i_forme]
        if x >= NW[0] and x < NE[0] and x >= SW[0] and x < SE[0] and y >= NW[1] and y < SW[1] and y >= NE[1] and y < SE[1] :
            return i_forme
    return -1

def generate_random_positions(nb_fourmis,nb_positions):
    fourmis_positions = []
    for i in range(nb_fourmis):
        positions = []
        for j in range(nb_positions):
            pos_x = np.random.randint(0,1080)
            pos_y = np.random.randint(0,1440) 
            positions.append((pos_x,pos_y))
        fourmis_positions.append(positions)
    return np.array(fourmis_positions)

# classe 1 : marche en vertical 
# classe -1 : marche en horizontal 
def generate_lineaire_positions(nb_fourmis,nb_positions):
    fourmis_positions = []
    pas_delacement = 10
    for i in range(int(nb_fourmis/2)):
        positions = []
        # fixer pour cette fourmi la ligne ou elle se deplace
        pos_x = np.random.randint(0,1080)
        for j in range(nb_positions):
            # faire varier sur l'axe Y pour simuler un deplacement sur cet axe
            pos_y = j*2 + pas_delacement
            positions.append((pos_x,pos_y))
        fourmis_positions.append(positions)
        
    for i in range(int(nb_fourmis/2)):
        positions = []
        # fixer pour cette fourmi la colonne ou elle se deplace
        pos_y = np.random.randint(0,1440)
        for j in range(nb_positions):
            # faire varier sur l'axe X pour simuler un deplacement sur cet axe
            pos_x = j*2 + pas_delacement
            positions.append((pos_x,pos_y))
        fourmis_positions.append(positions)
    return np.array(fourmis_positions)

def get_bag_of_positions(fourmis_positions,formes_grille,bins_h,bins_w):
    nb_fourmis = fourmis_positions.shape[0]
    matr_spr = np.zeros((nb_fourmis,bins_h*bins_w),dtype=int)
    for i_fourmis in range(len(fourmis_positions)):
        for posi in fourmis_positions[i_fourmis]:
            indice = indi_position_in_forme(formes_grille,posi)
            if indice != -1:
                matr_spr[i_fourmis][indice] += 1
    return matr_spr

def get_plot_fourmi(fourmi):
    X = fourmi[:,0]
    Y = fourmi[:,1]
    plt.scatter(X,Y)
    plt.show()
    
    
    
###############################################  
################ KALMAN FILTER ################  
###############################################  
    
    
    
    
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
    img = cv2.imread(bg)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    for pos in positions:
        cv2.circle(img,pos,15,(0,20,220),2)

    cv2.namedWindow(bg+"_test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(bg+"_test", 1500, 1000)    
    cv2.imshow(bg+"_test",img)
    cv2.waitKey(0)