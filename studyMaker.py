import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.animation
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
writergif = matplotlib.animation.PillowWriter(fps=3)

from arakawa import arakawa_grids, SWEsolver

# Constants
g = 9.81 # m/s^2
dx = 0.1 # m
dy = 0.1 # m
dt = 0.01 # s

xmax = 5 # m
ymax = 5 # m
nt = 301 

# Study parameters
# window = 1.0 # window \in [0,1] percentage of grid to animate around center point
window = 0.1
# initial = "gauss"
initial = "delta"
f = 1e-4 # s^-1
# f = 0. # s^-1

# Creating grids ===================================================================
x = np.linspace(-xmax, xmax, int(2.*xmax/dx)+1, endpoint=True)
y = np.linspace(-ymax, ymax, int(2.*ymax/dy)+1, endpoint=True)
refFactor = 5.
xref = np.linspace(-xmax, xmax, int(2.*refFactor*xmax/dx)+1, endpoint=True)
yref = np.linspace(-ymax, ymax, int(2.*refFactor*ymax/dy)+1, endpoint=True)
xref2 = np.linspace(-xmax, xmax, int(4.*refFactor*xmax/dx)+1, endpoint=True)
yref2 = np.linspace(-ymax, ymax, int(4.*refFactor*ymax/dy)+1, endpoint=True)
# print(x-xref[::int(refFactor)])
# print(x-xref2[::int(2*refFactor)])

X, Y = np.meshgrid(x, y)
U, V, H = np.zeros((3, len(x), len(y)))
XR, YR = np.meshgrid(xref, yref)
UR, VR, HR = np.zeros((3, len(xref), len(yref)))
XR2, YR2 = np.meshgrid(xref2, yref2)
UR2, VR2, HR2 = np.zeros((3, len(xref2), len(yref2)))

if initial == "gauss":
    sigma = 2.*dx
    H = (0.5/(sigma*np.sqrt(2*np.pi)))*np.exp(-(X**2+Y**2)/(2.*sigma**2))
    HR = (0.5/(sigma*np.sqrt(2*np.pi)))*np.exp(-(XR**2+YR**2)/(2.*sigma**2))
    HR2 = (0.5/(sigma*np.sqrt(2*np.pi)))*np.exp(-(XR2**2+YR2**2)/(2.*sigma**2))
elif initial == "delta":
    H[np.argwhere(x==0.)[0,0]:np.argwhere(x>=dx)[0,0],np.argwhere(y==0.)[0,0]:np.argwhere(y>=dy)[0,0]] = 1.0
    HR[np.argwhere(xref>=0.)[0,0]:np.argwhere(xref>=dx)[0,0],np.argwhere(yref>=0.)[0,0]:np.argwhere(yref>=dy)[0,0]] = 1.0
    HR2[np.argwhere(xref2>=0.)[0,0]:np.argwhere(xref2>=dx)[0,0],np.argwhere(yref2>=0.)[0,0]:np.argwhere(yref2>=dy)[0,0]] = 1.0
print("h0 = "+str(H[len(x)//2,len(y)//2]))

# Calculating time evolution ==================================================================================
print("calculating")
keys = list(arakawa_grids.keys())
solvers = [ SWEsolver(arakawa_grids[key], dx, dy, g, f, D=1, eps=0, nonlinear=False) for key in keys ]

# Ref 2
gridsR2 = [[UR2,VR2,HR2]]
maximumsR2 = np.amax(gridsR2[-1], axis=(1,2))
solverR2 = SWEsolver(arakawa_grids["C"], dx/(2.*refFactor), dy/(2.*refFactor), g, f, D=1, eps=0, nonlinear=False)
for j in range(int(nt*2.*refFactor)):
    if j%(2.*refFactor) == 0.: gridsR2.append(solverR2.singleTimeStep(gridsR2[-1][0], gridsR2[-1][1], gridsR2[-1][2], dt/(2.*refFactor) ))
    else: gridsR2[-1] = solverR2.singleTimeStep(gridsR2[-1][0], gridsR2[-1][1], gridsR2[-1][2], dt/(2.*refFactor) )
    maximumsR2 = np.maximum(maximumsR2, np.amax(np.abs(gridsR2[-1]), axis=(1,2)))
gridsR2 = np.array(gridsR2)
# Ref 1
gridsR = [[UR,VR,HR]]
devsR = [[np.zeros(UR.shape) for i in range(3)]]
maximumsR = np.amax(gridsR[-1], axis=(1,2))
maximums_devR = np.amax(devsR[-1], axis=(1,2))
solverR = SWEsolver(arakawa_grids["C"], dx/refFactor, dy/refFactor, g, f, D=1, eps=0, nonlinear=False)
for j in range(int(nt*refFactor)):
    if j%refFactor == 0.: gridsR.append(solverR.singleTimeStep(gridsR[-1][0], gridsR[-1][1], gridsR[-1][2], dt/refFactor))
    else: 
        gridsR[-1] = solverR.singleTimeStep(gridsR[-1][0], gridsR[-1][1], gridsR[-1][2], dt/refFactor)
        if j%refFactor == refFactor-1:
            devsR.append( gridsR[-1]-gridsR2[int(j/refFactor)][:,::2,::2] )
            maximumsR = np.maximum(maximumsR, np.amax(np.abs(gridsR[-1]), axis=(1,2)))
            maximums_devR = np.maximum(maximums_devR, np.amax(np.abs(devsR[-1]), axis=(1,2)))
gridsR = np.array(gridsR)
devsR = np.array(devsR)
# Arakwa grids
grids = [ [[U,V,H]] for i in range(len(keys)) ]
devs = [ [[np.zeros(U.shape) for j in range(3)]] for i in range(len(keys)) ]
maximums = np.array([np.amax(grids[i][-1], axis=(1,2)) for i in range(len(keys))])
maximums_dev = np.array([np.amax(devs[i][-1], axis=(1,2)) for i in range(len(keys))])
for j in range(nt):
    for i in range(len(keys)):    
        grids[i].append( np.array(solvers[i].singleTimeStep(grids[i][-1][0], grids[i][-1][1], grids[i][-1][2], dt)) )
        devs[i].append( grids[i][-1]-gridsR2[j][:,::int(2.*refFactor),::int(2.*refFactor)] )
        maximums[i] = np.maximum(maximums[i], np.amax(np.abs(grids[i][-1]), axis=(1,2)))
        maximums_dev[i] = np.maximum(maximums_dev[i], np.amax(np.abs(devs[i][-1]), axis=(1,2)))
grids = np.array(grids)
devs = np.array(devs)

#### Plotting ##### =================================================================================================
print("Ploting")
#### Base Animation -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(3, len(keys)+2)
fig.set_size_inches((len(keys)+2)*2, 3*2)
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.05, top=0.92, wspace=0.5, hspace=0.3)

imgs = [[],[],[]]
tick_pos = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
xind_range = (len(x)//2-int(len(x)*window*0.5), len(x)//2+int(len(x)*window*0.5)+1)
print(xind_range)
yind_range = (len(y)//2-int(len(y)*window*0.5), len(y)//2+int(len(y)*window*0.5)+1)
xind = np.array(tick_pos*(xind_range[1]-xind_range[0]), dtype=int)
yind = np.array(tick_pos*(yind_range[1]-yind_range[0]), dtype=int)
for i in range(len(keys)):
    ax[0][i].set_title(list(keys)[i])
    for k in range(3):
        ax[k][i].set_xlabel(r'$x(m)$', fontsize=6)
        ax[k][i].set_ylabel(r'$y(m)$', fontsize=6)
        ax[k][i].set_xticks(xind)
        ax[k][i].set_xticklabels(np.round(x[xind+xind_range[0]], 2), fontsize=6)
        ax[k][i].set_yticks(yind)
        ax[k][i].set_yticklabels(np.round(y[yind+yind_range[0]], 2), fontsize=6)
        imgs[k].append( ax[(k+1)%3][i].imshow(grids[i,0,k,xind_range[0]:xind_range[1],yind_range[0]:yind_range[1]], interpolation="none", cmap="seismic", vmin=-maximums[i][k], vmax=maximums[i][k]) )                
        divider = make_axes_locatable(ax[k][i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-maximums[i][k], vmax=maximums[i][k]), cmap="seismic"), cax=cax, orientation='vertical')
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(5)

ax[1][0].set_ylabel("U\n\n"+r'$y(m)$', fontsize=6)
ax[2][0].set_ylabel("V\n\n"+r'$y(m)$', fontsize=6)
ax[0][0].set_ylabel("H\n\n"+r'$y(m)$', fontsize=6)
# Ref
xRind_range = (len(xref)//2-int(len(xref)*window*0.5), len(xref)//2+int(len(xref)*window*0.5)+1)
yRind_range = (len(yref)//2-int(len(yref)*window*0.5), len(yref)//2+int(len(yref)*window*0.5)+1)
xRind = np.array(tick_pos*(xRind_range[1]-xRind_range[0]), dtype=int)
yRind = np.array(tick_pos*(yRind_range[1]-yRind_range[0]), dtype=int)
ax[0][-2].set_title("C"+r'$\times$'+str(int(refFactor)))
for k in range(3):
    ax[k][-2].set_xlabel(r'$x(m)$', fontsize=6)
    ax[k][-2].set_ylabel(r'$y(m)$', fontsize=6)
    ax[k][-2].set_xticks(xRind)
    ax[k][-2].set_xticklabels(np.round(xref[xRind+xRind_range[0]], 2), fontsize=6)
    ax[k][-2].set_yticks(yRind)
    ax[k][-2].set_yticklabels(np.round(yref[yRind+yRind_range[0]], 2), fontsize=6)
    imgs[k].append( ax[(k+1)%3][-2].imshow(gridsR[0,k,xRind_range[0]:xRind_range[1],yRind_range[0]:yRind_range[1]], interpolation="none", cmap="seismic", vmin=-maximumsR[k], vmax=maximumsR[k]) )
    divider = make_axes_locatable(ax[k][-2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-maximumsR[k], vmax=maximumsR[k]), cmap="seismic"), cax=cax, orientation='vertical')
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(5)
# Ref 2
xR2ind_range = (len(xref2)//2-int(len(xref2)*window*0.5), len(xref2)//2+int(len(xref2)*window*0.5)+1)
yR2ind_range = (len(yref2)//2-int(len(yref2)*window*0.5), len(yref2)//2+int(len(yref2)*window*0.5)+1)
xR2ind = np.array(tick_pos*(xR2ind_range[1]-xR2ind_range[0]), dtype=int)
yR2ind = np.array(tick_pos*(yR2ind_range[1]-yR2ind_range[0]), dtype=int)
ax[0][-1].set_title("C"+r'$\times$'+str(int(2*refFactor)))
for k in range(3):
    ax[k][-1].set_xlabel(r'$x(m)$', fontsize=6)
    ax[k][-1].set_ylabel(r'$y(m)$', fontsize=6)
    ax[k][-1].set_xticks(xR2ind)
    ax[k][-1].set_xticklabels(np.round(xref2[xR2ind+xR2ind_range[0]], 2), fontsize=6)
    ax[k][-1].set_yticks(yR2ind)
    ax[k][-1].set_yticklabels(np.round(yref2[yR2ind+yR2ind_range[0]], 2), fontsize=6)
    imgs[k].append( ax[(k+1)%3][-1].imshow(gridsR2[0,k,xR2ind_range[0]:xR2ind_range[1],yR2ind_range[0]:yR2ind_range[1]], interpolation="none", cmap="seismic", vmin=-maximumsR2[k], vmax=maximumsR2[k]) )
    divider = make_axes_locatable(ax[k][-1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-maximumsR2[k], vmax=maximumsR2[k]), cmap="seismic"), cax=cax, orientation='vertical')
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(5)

def update(t):
    for i in range(len(keys)):
        for k in range(3):
            imgs[k][i].set_array(grids[i,t,k,xind_range[0]:xind_range[1],yind_range[0]:yind_range[1]])
    # Ref
    for k in range(3):
        imgs[k][-2].set_array(gridsR[t,k,xRind_range[0]:xRind_range[1],yRind_range[0]:yRind_range[1]])
    # Ref2
    for k in range(3):
        imgs[k][-1].set_array(gridsR2[t,k,xR2ind_range[0]:xR2ind_range[1],yR2ind_range[0]:yR2ind_range[1]])
    fig.suptitle(str(np.round(t*dt,2))+" s")

ani = matplotlib.animation.FuncAnimation(fig, func=update, frames=nt, repeat=False, interval=500)
ani.save("/project/bfys/kspenko/sandbox/nma/3naloga/"+initial+"/"+initial+"_f="+str(f)+"_zoom"+str(window)+"_animation.gif",writer=writergif)
plt.close()

#### Deviation Animation -----------------------------------------------------------------------------------------------
fig, ax = plt.subplots(3, len(keys)+1)
fig.set_size_inches((len(keys)+1)*2, 3*2)
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.05, top=0.92, wspace=0.5, hspace=0.3)

imgs = [[],[],[]]
for i in range(len(keys)):
    ax[0][i].set_title(list(keys)[i])
    for k in range(3):
        ax[k][i].set_xlabel(r'$x(m)$', fontsize=6)
        ax[k][i].set_ylabel(r'$y(m)$', fontsize=6)
        ax[k][i].set_xticks(xind)
        ax[k][i].set_xticklabels(np.round(x[xind+xind_range[0]], 2), fontsize=6)
        ax[k][i].set_yticks(yind)
        ax[k][i].set_yticklabels(np.round(y[yind+yind_range[0]], 2), fontsize=6)
        imgs[k].append( ax[(k+1)%3][i].imshow(devs[i,0,k,xind_range[0]:xind_range[1],yind_range[0]:yind_range[1]], interpolation="none", cmap="seismic", vmin=-maximums_dev[i][k], vmax=maximums_dev[i][k]) )
        divider = make_axes_locatable(ax[k][i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-maximums_dev[i][k], vmax=maximums_dev[i][k]), cmap="seismic"), cax=cax, orientation='vertical')
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(5)
ax[1][0].set_ylabel(r'$\Delta$'+"U\n\n"+r'$y(m)$', fontsize=6)
ax[2][0].set_ylabel(r'$\Delta$'+"V\n\n"+r'$y(m)$', fontsize=6)
ax[0][0].set_ylabel(r'$\Delta$'+"H\n\n"+r'$y(m)$', fontsize=6)
# Ref
ax[0][-1].set_title("C"+r'$\times$'+str(int(refFactor)))
for k in range(3):
    ax[k][-1].set_xlabel(r'$x(m)$', fontsize=6)
    ax[k][-1].set_ylabel(r'$y(m)$', fontsize=6)
    ax[k][-1].set_xticks(xRind)
    ax[k][-1].set_xticklabels(np.round(xref[xRind+xRind_range[0]], 2), fontsize=6)
    ax[k][-1].set_yticks(yRind)
    ax[k][-1].set_yticklabels(np.round(yref[yRind+yRind_range[0]], 2), fontsize=6)
    imgs[k].append( ax[(k+1)%3][-1].imshow(devsR[0,k,xRind_range[0]:xRind_range[1],yRind_range[0]:yRind_range[1]], interpolation="none", cmap="seismic", vmin=-maximums_devR[k], vmax=maximums_devR[k]) )
    divider = make_axes_locatable(ax[k][-1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-maximums_devR[k], vmax=maximums_devR[k]), cmap="seismic"), cax=cax, orientation='vertical')
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(5)

print(gridsR.shape)
print(devsR.shape)

def update(t):
    for i in range(len(keys)):
        for k in range(3):
            imgs[k][i].set_array(devs[i,t,k,xind_range[0]:xind_range[1],yind_range[0]:yind_range[1]])
    # Ref
    for k in range(3):
        imgs[k][-1].set_array(devsR[t,k,xRind_range[0]:xRind_range[1],yRind_range[0]:yRind_range[1]])
    fig.suptitle(str(np.round(t*dt,2))+" s")

ani = matplotlib.animation.FuncAnimation(fig, func=update, frames=nt, repeat=False, interval=500)
ani.save("/project/bfys/kspenko/sandbox/nma/3naloga/"+initial+"/"+initial+"_f="+str(f)+"_zoom"+str(window)+"_deviation.gif",writer=writergif)
plt.close()

#### Cross section -----------------------------------------------------------------------------------------------
crossSections = ["y=0","x=0"]
plots = ["U","V","H"]
fig, ax = plt.subplots(3, 2)
fig.set_size_inches(12, 8)
fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)

lines = [[[] for j in range(len(crossSections))] for k in range(3)]
for k in range(3):
    klim = np.amax( [maximumsR[k], maximumsR2[k]] + list(maximums[:,k]) )
    ax[0][0].set_title(crossSections[0])
    ax[(k+1)%3][0].set_xlabel(r'$x(m)$')
    ax[(k+1)%3][0].set_ylabel(plots[k])
    for i in range(len(keys)):
        lines[k][0].append( ax[(k+1)%3][0].plot(x[int(len(x)//2):xind_range[1]], grids[i,0,k,int(len(x)//2):xind_range[1],int(len(y)//2)], label=keys[i], linewidth=1)[0] )
    lines[k][0].append( ax[(k+1)%3][0].plot(xref[int(len(xref)//2):xRind_range[1]], gridsR[0,k,int(len(xref)//2):xRind_range[1],int(len(yref)//2)], label="C"+r'$\times$'+str(int(refFactor)), color="gray", linestyle=":", linewidth=1)[0] )
    lines[k][0].append( ax[(k+1)%3][0].plot(xref2[int(len(xref2)//2):xR2ind_range[1]], gridsR2[0,k,int(len(xref2)//2):xR2ind_range[1],int(len(yref2)//2)], label="C"+r'$\times$'+str(int(2*refFactor)), color="k", linestyle="--", linewidth=1)[0] )
    ax[(k+1)%3][0].legend(loc="upper right", prop={'size': 6})
    ax[(k+1)%3][0].set_ylim((-klim*1.1, klim*1.1))

    ax[0][1].set_title(crossSections[1])
    ax[(k+1)%3][1].set_xlabel(r'$y(m)$')
    ax[(k+1)%3][1].set_ylabel(plots[k])
    for i in range(len(keys)):
        lines[k][1].append( ax[(k+1)%3][1].plot(y[int(len(y)//2):yind_range[1]], grids[i,0,k,int(len(x)//2),int(len(y)//2):yind_range[1]], label=keys[i], linewidth=1)[0] )
    lines[k][1].append( ax[(k+1)%3][1].plot(yref[int(len(yref)//2):yRind_range[1]], gridsR[0,k,int(len(xref)//2),int(len(yref)//2):yRind_range[1]], label="C"+r'$\times$'+str(int(refFactor)), color="gray", linestyle=":", linewidth=1)[0] )
    lines[k][1].append( ax[(k+1)%3][1].plot(yref2[int(len(yref2)//2):yR2ind_range[1]], gridsR2[0,k,int(len(xref2)//2),int(len(yref2)//2):yR2ind_range[1]], label="C"+r'$\times$'+str(int(2*refFactor)), color="k", linestyle="--", linewidth=1)[0] )
    ax[(k+1)%3][1].legend(loc="upper right", prop={'size': 6})
    ax[(k+1)%3][1].set_ylim((-klim*1.1, klim*1.1))

def update(t):
    for k in range(3):
        for i in range(len(keys)):
            lines[k][0][i].set_ydata(grids[i,t,k,int(len(x)//2):xind_range[1],int(len(y)//2)])
            lines[k][1][i].set_ydata(grids[i,t,k,int(len(x)//2),int(len(y)//2):yind_range[1]])
        # Ref
        lines[k][0][-2].set_ydata(gridsR[t,k,int(len(xref)//2):xRind_range[1],int(len(yref)//2)])
        lines[k][1][-2].set_ydata(gridsR[t,k,int(len(xref)//2),int(len(yref)//2):yRind_range[1]])
        # Ref2
        lines[k][0][-1].set_ydata(gridsR2[t,k,int(len(xref2)//2):xR2ind_range[1],int(len(yref2)//2)])
        lines[k][1][-1].set_ydata(gridsR2[t,k,int(len(xref2)//2),int(len(yref2)//2):yR2ind_range[1]])
    fig.suptitle(str(np.round(t*dt,2))+" s")

ani = matplotlib.animation.FuncAnimation(fig, func=update, frames=nt, repeat=False, interval=500)
ani.save("/project/bfys/kspenko/sandbox/nma/3naloga/"+initial+"/"+initial+"_f="+str(f)+"_zoom"+str(window)+"_crossSection.gif",writer=writergif)
plt.close()

#### Cross Deviations -----------------------------------------------------------------------------------------------
crossSections = ["y=0","x=0"]
plots = ["U","V","H"]
fig, ax = plt.subplots(3, 2)
fig.set_size_inches(12, 8)
fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)

lines = [[[] for j in range(len(crossSections))] for k in range(3)]
for k in range(3):
    klim = np.amax( [maximums_devR[k]] + list(maximums_dev[:,k]) )
    ax[0][0].set_title(crossSections[0])
    ax[(k+1)%3][0].set_xlabel(r'$x(m)$')
    ax[(k+1)%3][0].set_ylabel(r'$\Delta$'+plots[k])
    for i in range(len(keys)):
        lines[k][0].append( ax[(k+1)%3][0].plot(x[int(len(x)//2):xind_range[1]], devs[i,0,k,int(len(x)//2):xind_range[1],int(len(y)//2)], label=keys[i], linewidth=1)[0] )
    lines[k][0].append( ax[(k+1)%3][0].plot(xref[int(len(xref)//2):xRind_range[1]], devsR[0,k,int(len(xref)//2):xRind_range[1],int(len(yref)//2)], label="C"+r'$\times$'+str(int(refFactor)), color="gray", linestyle=":", linewidth=1)[0] )
    ax[(k+1)%3][0].legend(loc="upper right", prop={'size': 6})
    ax[(k+1)%3][0].set_ylim((-klim*1.1, klim*1.1))

    ax[0][1].set_title(crossSections[1])
    ax[(k+1)%3][1].set_xlabel(r'$y(m)$')
    ax[(k+1)%3][1].set_ylabel(r'$\Delta$'+plots[k])
    for i in range(len(keys)):
        lines[k][1].append( ax[(k+1)%3][1].plot(y[int(len(y)//2):yind_range[1]], devs[i,0,k,int(len(x)//2),int(len(y)//2):yind_range[1]], label=keys[i], linewidth=1)[0] )
    lines[k][1].append( ax[(k+1)%3][1].plot(yref[int(len(yref)//2):yRind_range[1]], devsR[0,k,int(len(xref)//2),int(len(yref)//2):yRind_range[1]], label="C"+r'$\times$'+str(int(refFactor)), color="gray", linestyle=":", linewidth=1)[0] )
    ax[(k+1)%3][1].legend(loc="upper right", prop={'size': 6})
    ax[(k+1)%3][1].set_ylim((-klim*1.1, klim*1.1))

def update(t):
    for k in range(3):
        for i in range(len(keys)):
            lines[k][0][i].set_ydata(devs[i,t,k,int(len(x)//2):xind_range[1],int(len(y)//2)])
            lines[k][1][i].set_ydata(devs[i,t,k,int(len(x)//2),int(len(y)//2):yind_range[1]])
        # Ref
        lines[k][0][-1].set_ydata(devsR[t,k,int(len(xref)//2):xRind_range[1],int(len(yref)//2)])
        lines[k][1][-1].set_ydata(devsR[t,k,int(len(xref)//2),int(len(yref)//2):yRind_range[1]])
    fig.suptitle(str(np.round(t*dt,2))+" s")

ani = matplotlib.animation.FuncAnimation(fig, func=update, frames=nt, repeat=False, interval=500)
ani.save("/project/bfys/kspenko/sandbox/nma/3naloga/"+initial+"/"+initial+"_f="+str(f)+"_zoom"+str(window)+"_crossDeviations.gif",writer=writergif)
plt.close()