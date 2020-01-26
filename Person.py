from random import randint
import time
class MyPerson:
    tracks = []
    def __init__(self, xi, yi):
        self.x = xi
        self.y = yi
        self.tracks = []
        self.done = False
        self.state = '0'        # 0 -if object did not crossed the line, 1 - if yes
        #self.dir = None
    def getTracks(self):
        return self.tracks
   # def getDir(self):
       # return self.dir
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x,self.y])
        self.x = xn
        self.y = yn
    def timedOut(self):
        return self.done
    # Check if the centroid has crossed the ref. line
    def UP(self,top,bottom):
        if len(self.tracks) >= 4:
            if self.state == '0':
                if self.tracks[-1][1] < top and self.tracks[-2][1] > top:
                    self.state = '1'
                    self.done = True
                    #self.dir = 'out'
                    return True
            else:
                return False
        else:
            return False


    def DOWN(self,top,bottom):
        if len(self.tracks) >= 4:
            if self.state == '0':
                if self.tracks[-1][1] > bottom and self.tracks[-2][1] < bottom:
                    self.state = '1'
                    self.done = True
                    #self.dir = 'in'
                    return True
            else:
                return False
        else:
            return False
