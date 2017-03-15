class Multiply(object):
    """docstring for """
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def forward(self):
        z=self.x*self.y
        return z

    def backward(self,dz):
        dx=self.y*dz
        dy=self.x*dz
        return [dx,dy]

class Add(object):
    """docstring for AddGate"""
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def forward(self):
        z=self.x+self.y
        return z

    def backward(self,dz):
        return [dz,dz]

class Max(object):
    """docstring for Max"""
    def forward(self,x,y):
        m=max(self.x,self.y)
        self.wasitx = (m==self.x)
        return m

    def backward(self,dz):
        if(self.wasitx) : return [dz,0]
        else : return [0,dz]
