import math
import numpy as np

def dist(pos1, pos2):
    return np.linalg.norm(pos1-pos2)

def trouvePlusProche(f0, listFourmis, seuil=30):
    id = -1
    distances = [dist(f0.getPos(), np.array(data[-1][0], data[-1][1])) for _, data in listFourmis.items()]
    if min(distances) < seuil:
        id = distances.index(min(distances))
    return id

class Fourmis():
    def __init__(self, id, x, y, w, l, velocite = None):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.l = l
        self.centre = np.array([x + w/2,y + l/2])
        self.velocite = velocite
        
    def getPos(self):
        return self.centre
    
    def getVelocite(self):
        return self.velocite
    
    def getData(self):
        return self.centre,self.velocite

    def getId(self):
        return self.id
    
    
class GestionFourmis():
    def __init__(self):
        self.ListeFourmis = {}
        
    def add(self, fourmis):
        if fourmis.getId() not in self.ListeFourmis.keys():
            self.ListeFourmis[fourmis.getId()] = [fourmis.getData()]
        else:
            self.ListeFourmis[fourmis.getId()].append(fourmis.getData())
        
    def getNbFourmis(self):
        return len(self.ListeFourmis)
    
    def majFourmis(self, id, data):
        self.ListeFourmis[id].append(data)
        
    def getListe(self):
        return self.ListeFourmis