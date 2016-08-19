class EntryVehicle:
    def __init__(self, mass = 2804.0, area = 15.8, CD = 0, CL = 0):
        self.mass = mass
        self.area = area
        self.CD = CD
        self.CL = CL

        
    def aerodynamic_coefficients(self, M):
        pD = [2.598e4, -1022.0, -2904.0, 678.6, -44.33, 1.373]
        qD = [1.505e4, 1687.0, -2651.0, 544.1, -34.11, 1]
        pL = [1.172e4, -3654.0, 485.6, -14.61, 0.4192]
        qL = [2.53e4, -7846.0, 1086.0, -28.35, 1]
        
        num, den = 0, 0
        for i in range(0, len(pD)):
            num = num + pD[i]*M**i   
        for i in range(0, len(qD)):
            den = den + qD[i]*M**i
            
        cD = num/den
        
        num, den = 0, 0
        for i in range(0, len(pL)):
            num = num + pL[i]*M**i
            
        for i in range(0, len(qL)):
            den = den + qL[i]*M**i
            
        cL = num/den
        return cD*(1+self.CD), cL*(1+self.CL)
