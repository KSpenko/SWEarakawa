import numpy as np  
import matplotlib.pyplot as plt  
    
    

    
def idw2d(xi, yi, zi, xf, yf):
    ri = np.sqrt((xi-xf)**2+(yi-yf)**2)
    zero_ind = np.argwhere(ri < 1e-3)
    if len(zero_ind > 0): return zi[zero_ind[0]][0]

    sum1 = 0 
    sum2 = 0
    for i in range(len(ri)):
        if ri[i] >= 1: continue
        sum1 += ri[i]
        sum2 += ri[i]*zi[i]
    return sum2/sum1

#=========================================================================================    
if __name__ == "__main__": 
    sigma = 2.
    a = np.arange(-10, 10, 0.2)  
    b = np.arange(-10, 10, 0.2)  
    x, y = np.meshgrid(a, b)  
    z = np.sin((x**2. + (y-5)**2.)/(2.*sigma**2.))

    fig, ax = plt.subplots(1,2)
    h = ax[0].imshow(z)  

    n = len(np.ndarray.flatten(x))
    n = int(n/5)
    inda = np.random.randint(0, len(a), n)
    indb = np.random.randint(0, len(b), n)
    uf = idw2d( x[inda,indb], y[inda,indb], z[inda,indb], 0.123123123123123123, 0.123121423123123)

    u = []
    for i in range(len(a)):
        u.append([])
        for j in range(len(b)):
            u[-1].append( idw2d( x[inda,indb], y[inda,indb], z[inda,indb], x[i,j], y[i,j]) )

    h = ax[1].imshow(np.array(u), vmin=np.amin(z), vmax=np.amax(z))  
    plt.savefig("idw.png")