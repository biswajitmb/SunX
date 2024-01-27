from scipy.io import readsav
import matplotlib.pyplot as plt
import numpy as np

sav_data = readsav('t.sav')#read idl output

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
ax2.plot_surface(x, y, bcube[:,:,10,0])
'''
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(x, y, sav_data['bcube'][0,10,:,:])
#ax.plot_surface(X, Y, Z_pred)
plt.show()
'''

tt=readsav('tt.sav')
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(tt['x'], tt['y'], sav_data['bcube'][0,10,:,:])
#ax.plot_surface(X, Y, Z_pred)
plt.show()
