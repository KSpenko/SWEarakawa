import numpy as np
import matplotlib.pyplot as plt

# Unable to show include E because it has a bigger desnity of points!
arakawa_grids = { # h=(0,0) and u(x,y), v(x,y) relative coordinates always positive x_u, y_u, x_v, y_v \in [0,1]!
    "A": ((0,0), (0,0)),
    "B": ((0.5,0.5), (0.5,0.5)),
    "C": ((0.5,0), (0,0.5)),
    "D": ((0,0.5), (0.5,0)),
    "KS": ((0.2,0.3), (0.8,0.7)),
}

class SWEsolver:
    def __init__(self, arakawa, dx, dy, g=1, f=1, D=1, eps=0, nonlinear=False):
        self.arakawa = arakawa
        self.dx = dx
        self.dy = dy
        self.g = g
        self.f = f
        self.D = D
        self.eps = eps
        self.nonlinear = nonlinear

        self.uv_dx = (self.arakawa[0][0]-self.arakawa[1][0])*self.dx
        self.uv_dy = (self.arakawa[0][1]-self.arakawa[1][1])*self.dy
        self.uh_dx = (self.arakawa[0][0])*self.dx
        self.uh_dy = (self.arakawa[0][1])*self.dy
        self.vh_dx = (self.arakawa[1][0])*self.dx
        self.vh_dy = (self.arakawa[1][1])*self.dy
        self.uv_sx = int(np.sign(self.uv_dx))
        self.uv_sy = int(np.sign(self.uv_dy))
        self.uh_sx = int(np.sign(self.uh_dx))
        self.uh_sy = int(np.sign(self.uh_dy))
        self.vh_sx = int(np.sign(self.vh_dx))
        self.vh_sy = int(np.sign(self.vh_dy))


    def SWE_dUdt(self, U, V, H):
        # U, V, H are all 2D matrices with shape (n,m)
        n, m = U.shape
        dUdt = np.zeros((n, m))

        # 1st equation for U
        if self.nonlinear:
            dUdx = (U[2:,1:-1]-U[:-2,1:-1])/(2.*self.dx)
            dUdy = (U[1:-1,2:]-U[1:-1,:-2])/(2.*self.dy)

        VatU = (self.dx-self.uv_sx*self.uv_dx)*(self.dy-self.uv_sy*self.uv_dy)*V[1:-1,1:-1] 
        VatU += (self.uv_sx*self.uv_dx)*(self.dy-self.uv_sy*self.uv_dy)*V[1+self.uv_sx:n-1+self.uv_sx,1:-1] 
        VatU += (self.dx-self.uv_sx*self.uv_dx)*(self.uv_sy*self.uv_dy)*V[1:-1,1+self.uv_sy:m-1+self.uv_sy] 
        VatU += (self.uv_sx*self.uv_dx)*(self.uv_sy*self.uv_dy)*V[1+self.uv_sx:n-1+self.uv_sx,1+self.uv_sy:m-1+self.uv_sy]
        VatU /= (self.dx*self.dy)
        # VatU = V[1:-1,1:-1] 

        dHdx = (self.dy-self.uh_dy)*(H[2:,1:-1] - H[self.uh_sx:n-2+self.uh_sx,1:-1])
        dHdx += self.uh_dy*(H[2:,1+self.uh_sy:m-1+self.uh_sy] - H[self.uh_sx:n-2+self.uh_sx,1+self.uh_sy:m-1+self.uh_sy])
        dHdx /= (self.dy*self.dx*(2-self.uh_sx))
        # dHdx = 0.5*(H[2:,1:-1] - H[:-2,1:-1])/self.dx 
        # dHdx = 0.5*(H[2:,2:]-H[1:-1,2:]+H[2:,1:-1]-H[1:-1,1:-1])/self.dx
        # dHdx = 0.5*(H[1:-1,1:-1]-H[0:-2,1:-1]+H[1:-1,0:-2]-H[0:-2,0:-2])/self.dx
        
        dUdt[1:-1,1:-1] = np.multiply(VatU, self.f) - self.g*dHdx - np.multiply(U[1:-1,1:-1], self.eps)
        if self.nonlinear: dUdt[1:-1,1:-1] -= np.multiply(U[1:-1,1:-1], dUdx) + np.multiply(VatU, dUdy) 
        return dUdt

    def SWE_dVdt(self, U, V, H):
        # U, V, H are all 2D matrices with shape (n,m)
        n, m = U.shape
        dVdt = np.zeros((n, m))

        # 2nd equation for V
        if self.nonlinear:
            dVdx = (V[2:,1:-1]-V[:-2,1:-1])/(2.*self.dx)
            dVdy = (V[1:-1,2:]-V[1:-1,:-2])/(2.*self.dy)

        UatV = (self.dx-self.uv_sx*self.uv_dx)*(self.dy-self.uv_sy*self.uv_dy)*U[1:-1,1:-1] 
        UatV += (self.uv_sx*self.uv_dx)*(self.dy-self.uv_sy*self.uv_dy)*U[1-self.uv_sx:n-1-self.uv_sx,1:-1] 
        UatV += (self.dx-self.uv_sx*self.uv_dx)*(self.uv_sy*self.uv_dy)*U[1:-1,1-self.uv_sy:m-1-self.uv_sy] 
        UatV += (self.uv_sx*self.uv_dx)*(self.uv_sy*self.uv_dy)*U[1-self.uv_sx:n-1-self.uv_sx,1-self.uv_sy:m-1-self.uv_sy]
        UatV /= (self.dx*self.dy)
        # UatV = U[1:-1,1:-1] 

        dHdy = (self.dx-self.vh_dx)*(H[1:-1,2:] - H[1:-1,self.vh_sy:m-2+self.vh_sy])
        dHdy += self.vh_dx*(H[1+self.vh_sx:n-1+self.vh_sx,2:] - H[1+self.vh_sx:n-1+self.vh_sx,self.vh_sy:m-2+self.vh_sy])
        dHdy /= (self.dx*self.dy*(2-self.vh_sy))
        # dHdy = 0.5*(H[1:-1,2:] - H[1:-1,:-2])/self.dy
        # dHdy = 0.5*(H[2:,2:]-H[2:,1:-1]+H[1:-1,2:]-H[1:-1,1:-1])/self.dy
        # dHdy = 0.5*(H[1:-1,1:-1]-H[1:-1,0:-2]+H[0:-2,1:-1]-H[0:-2,0:-2])/self.dy

        dVdt[1:-1,1:-1] = -np.multiply(UatV, self.f) - self.g*dHdy - np.multiply(V[1:-1,1:-1], self.eps)
        if self.nonlinear: dVdt[1:-1,1:-1] -= np.multiply(UatV, dVdx) + np.multiply(V[1:-1,1:-1], dVdy) 
        return dVdt

    def SWE_dHdt(self, U, V, H):
        # U, V, H are all 2D matrices with shape (n,m)
        n, m = U.shape
        dHdt = np.zeros((n, m))

        # 3rd equation for H
        if self.nonlinear:
            dHdx = (H[2:,1:-1]-H[:-2,1:-1])/(2.*self.dx)
            dHdy = (H[1:-1,2:]-H[1:-1,:-2])/(2.*self.dy)

            UatH = (self.dx-self.uh_dx)*(self.dy-self.uh_dy)*U[1:-1,1:-1] 
            UatH += (self.uh_dx)*(self.dy-self.uh_dy)*U[0:-2,1:-1] 
            UatH += (self.dx-self.uh_dx)*(self.uh_dy)*U[1:-1,0:-2] 
            UatH += (self.uh_dx)*(self.uh_dy)*U[0:-2,0:-2]
            UatH /= (self.dx*self.dy)
            # UatH = U[1:-1,1:-1] 

            VatH = (self.dx-self.vh_dx)*(self.dy-self.vh_dy)*V[1:-1,1:-1] 
            VatH += (self.vh_dx)*(self.dy-self.vh_dy)*V[0:-2,1:-1] 
            VatH += (self.dx-self.vh_dx)*(self.vh_dy)*V[1:-1,0:-2] 
            VatH += (self.vh_dx)*(self.vh_dy)*V[0:-2,0:-2]
            VatH /= (self.dx*self.dy)
            # VatH = V[1:-1,1:-1] 

        dUdx = (self.dy-self.uh_dy)*(U[2-self.uh_sx:n-self.uh_sx,1:-1] - U[0:-2,1:-1])
        dUdx += self.uh_dy*(U[2-self.uh_sx:n-self.uh_sx,1-self.uh_sy:m-1-self.uh_sy] - U[0:-2,1-self.uh_sy:m-1-self.uh_sy])
        dUdx /= (self.dy*self.dx*(2-self.uh_sx))
        # dUdx = 0.5*(U[2:,1:-1]-U[:-2,1:-1])/self.dx 
        # dUdx = 0.5*(U[1:-1,1:-1]-U[0:-2,1:-1]+U[1:-1,0:-2]-U[0:-2,0:-2])/self.dx
        # dUdx = 0.5*(U[2:,1:-1]-U[1:-1,1:-1]+U[2:,2:]-U[1:-1,2:])/self.dx

        dVdy = (self.dx-self.vh_dx)*(V[1:-1,2-self.vh_sy:m-self.vh_sy] - V[1:-1,0:-2])
        dVdy += self.vh_dx*(V[1-self.vh_sx:n-1-self.vh_sx,2-self.vh_sy:m-self.vh_sy] - V[1-self.vh_sx:n-1-self.vh_sx,0:-2])
        dVdy /= (self.dx*self.dy*(2-self.vh_sy))
        # dVdy = 0.5*(V[1:-1,2:]-V[1:-1,:-2])/self.dy 
        # dVdy = 0.5*(V[1:-1,1:-1]-V[1:-1,0:-2]+V[0:-2,1:-1]-V[0:-2,0:-2])/self.dy
        # dVdy = 0.5*(V[1:-1,2:]-V[1:-1,1:-1]+V[2:,2:]-V[2:,1:-1])/self.dy

        dHdt[1:-1,1:-1] = -np.multiply(H[1:-1,1:-1]+self.D, self.eps+dUdx+dVdy) 
        if self.nonlinear: dHdt[1:-1,1:-1] -= np.multiply(UatH, dHdx) + np.multiply(VatH, dHdy)
        return dHdt

    def singleTimeStep(self, U0, V0, H0, dt):
        U = np.copy(U0)
        V = np.copy(V0)
        H = np.copy(H0)
        U += dt*self.SWE_dUdt(U, V, H)
        V += dt*self.SWE_dVdt(U, V, H)
        H += dt*self.SWE_dHdt(U, V, H)
        return U, V, H

#============================================================================================================
if __name__ == "__main__":
    # constants
    g = 1
    f = 1

    # Setup grid
    dx = 0.1
    x = np.linspace(-1, 1, int(2/dx))
    dy = 0.1
    y = np.linspace(-1, 1, int(2/dy))
    n = len(x)

    # setup timestepping
    dt = 0.05
    m = 10

    X, Y = np.meshgrid(x, y)
    U, V, H = np.zeros((3, n, n))
    # sigma = 2.*delta
    # H = (10./(sigma*np.sqrt(2*np.pi)))*np.exp(-(X**2+Y**2)/(2.*sigma**2))
    H[n//2,n//2] = 0.5

    solver = SWEsolver(arakawa_grids["KS"], dx, dy, g, f, D=1, eps=0, nonlinear=False)
    
    maximums = [0,0,0]
    for i in range(m):
        fig, ax = plt.subplots(1,3)
        print(np.amax(np.abs(H)))
        ax[0].imshow(U, vmin=-0.25, vmax=0.25, cmap="bwr")
        ax[1].imshow(V, vmin=-0.25, vmax=0.25, cmap="bwr")
        ax[2].imshow(H, vmin=-0.5, vmax=0.5, cmap="bwr")
        temp_max = [np.amax(np.abs(U)), np.amax(np.abs(V)), np.amax(np.abs(H))]
        maximums = np.maximum(maximums, temp_max)
        plt.savefig("swe"+str(i)+".png")
        plt.show()
        U, V, H = solver.singleTimeStep(U, V, H, dt)
    # plt.close()
    print(maximums)