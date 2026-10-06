import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import readsav

plot_TR = True

# ============================================================
# READ DATA
# ============================================================

idl_file = (
    '/Users/bmondal/BM_Works/softwares/ssw/packages/'
    'GX_SIMULATOR/euv/ebtel/ebtel_scale=0.2_alpha=-2.5.sav'
)

new_file = './outputs/ebtel_table_LognormalEnergyDistribution_RandomDelay.sav'

da = readsav(idl_file)
da_new = readsav(new_file)


# ------------------------------------------------------------
# Old data
# ------------------------------------------------------------

lrun = da['lrun']
qrun = da['qrun']

L_all = lrun.flatten()
Vol_Q_all = qrun.flatten()


# ------------------------------------------------------------
# New data
# ------------------------------------------------------------

lrun_new = da_new['lrun']
qrun_new = da_new['qrun']

L_all_new = lrun_new.flatten()
Vol_Q_all_new = qrun_new.flatten()


# ============================================================
# DIMENSIONS
# ============================================================

ny, nx = lrun.shape

print("Grid shape =", lrun.shape)

print("Old DEM shape =", da['DDM_COR_RUN'].shape)
print("New DEM shape =", da_new['DDM_COR_RUN'].shape)

plt.close('all')
# ============================================================
# TEMPERATURE ARRAYS
# ============================================================

logt_old = np.asarray(da['LOGTDEM']).squeeze()
logt_new = np.asarray(da_new['LOGTDEM']).squeeze()

l_ind_all = range(ny)
#l_ind_all = [8]

for l_ind in l_ind_all:
    #for q_ind in range(nx):
    for Q in [0.001]:
        q_ind = np.where(abs(qrun[l_ind,:] - Q) == abs(qrun[l_ind,:] - Q).min())[0][0]

        # ============================================================
        # FIGURE
        # ============================================================
        
        fig, (ax_map, ax_dem) = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            constrained_layout=True
        )
        
        
        # ============================================================
        # LEFT PANEL: L versus Q
        # ============================================================
        
        ax_map.plot(
            L_all,
            Vol_Q_all,
            '+',
            color='k',
            label='Old'
        )
        
        ax_map.plot(
            L_all_new,
            Vol_Q_all_new,
            '+',
            color='r',
            alpha=0.1,
            label='New'
        )
        
        ax_map.set_xscale('log')
        ax_map.set_yscale('log')
        
        ax_map.set_xlabel('Loop Length (cm)')
        ax_map.set_ylabel(
            r'Volumetric $\langle Q\rangle$ '
            r'(erg cm$^{-3}$ s$^{-1}$)'
        )
        
        ax_map.legend()
        
        ax_map.set_title('Old (black) and New (red) Grids')
        
        
        # ------------------------------------------------------------
        # Marker for currently selected point
        # ------------------------------------------------------------

        selected_point, = ax_map.plot(
            [],
            [],
            'o',
            markersize=10,
            markerfacecolor='none',
            markeredgecolor='b',
            markeredgewidth=2
        )


        # ============================================================
        # RIGHT PANEL: DEM
        # ============================================================
        
        dem_line_cor_old, = ax_dem.step(
            [],
            [],
            where='mid',
            lw=2,
            linestyle='--',
            color='r',
            label='Old Cor'
        )
        
        dem_line_cor_new, = ax_dem.step(
            [],
            [],
            where='mid',
            lw=2,
            color='r',
            label='New Cor'
        )
        if plot_TR:
            dem_line_tr_old, = ax_dem.step(
                [],
                [],
                where='mid',
                lw=2,
                linestyle='--',
                color='b',
                label='Old TR'
            )
            
            dem_line_tr_new, = ax_dem.step(
                [],
                [],
                where='mid',
                lw=2,
                color='b',
                label='New TR'
            )
        
        ax_dem.set_yscale('log')
        
        ax_dem.set_xlabel('log T')
        ax_dem.set_ylabel('DEM (cm$^{-5}$ K$^{-1}$)')
        
        ax_dem.legend()
        
        ax_dem.set_title('DEM')
        
        last_index = [None, None]       
 
        # ============================================================
        # PRECOMPUTE LOG COORDINATES
        # ============================================================
        #
        # Because both axes are logarithmic, nearest point should be
        # calculated in log-space rather than ordinary linear space.
        #
        # Otherwise high-L/high-Q values dominate the distance.
        # ============================================================
        
        valid = (
            np.isfinite(lrun)
            & np.isfinite(qrun)
            & (lrun > 0)
            & (qrun > 0)
        )
        
        logL_grid = np.full_like(lrun, np.nan, dtype=float)
        logQ_grid = np.full_like(qrun, np.nan, dtype=float)
        
        logL_grid[valid] = np.log10(lrun[valid])
        logQ_grid[valid] = np.log10(qrun[valid])
        
        
        mouse_logL = np.log10(lrun[l_ind,q_ind])
        mouse_logQ = np.log10(qrun[l_ind,q_ind])

        
        # --------------------------------------------------------
        # Find closest old-data point
        # --------------------------------------------------------

        distance2 = (
            (logL_grid - mouse_logL)**2
            +
            (logQ_grid - mouse_logQ)**2
        )

        if np.all(~np.isfinite(distance2)):break

        flat_index = np.nanargmin(distance2)

        iy, ix = np.unravel_index(
            flat_index,
            lrun.shape
        )
        
        iy, ix = l_ind,q_ind
        # --------------------------------------------------------
        # Do nothing if still on same point
        # --------------------------------------------------------

        if (
            iy == last_index[0]
            and
            ix == last_index[1]
        ):
            break

        last_index[0] = iy
        last_index[1] = ix

        L_selected = lrun[iy, ix]
        Q_selected = qrun[iy, ix]

        #Find new data index:
        # Distance in log space
        
        distance2_new = (
            (np.log10(lrun_new) - np.log10(L_selected))**2
            +
            (np.log10(qrun_new) - np.log10(Q_selected))**2
        )
        
        flat_index_new = np.nanargmin(distance2_new)
        
        iy_new, ix_new = np.unravel_index(
            flat_index_new,
            lrun_new.shape
        )
        
        #iy_new=46
        
        # ========================================================
        # GET DEM
        # ========================================================

        dem_cor_pix_old = np.asarray(
            da['DEM_COR_RUN'][iy, ix, :]
        ).squeeze() * lrun[iy, ix]

        dem_tr_pix_old = np.asarray(
            da['DEM_TR_RUN'][iy, ix, :]
        ).squeeze() * 2

        dem_cor_pix_new = np.asarray(
            da_new['DEM_COR_RUN'][iy_new, ix_new, :]
        ).squeeze() * lrun_new[iy_new, ix_new]

        dem_tr_pix_new = np.asarray(
            da_new['DEM_TR_RUN'][iy_new, ix_new, :]
        ).squeeze() * 2

        print(lrun[iy, ix],qrun[iy, ix])
        print(lrun_new[iy_new, ix_new],qrun_new[iy_new, ix_new])
        print('=='*10)

        # ========================================================
        # UPDATE DEM LINES
        # ========================================================

        dem_line_cor_old.set_data(
            logt_old,
            dem_cor_pix_old
        )

        dem_line_cor_new.set_data(
            logt_new,
            dem_cor_pix_new
        )

        if plot_TR:
            dem_line_tr_old.set_data(
                logt_old,
                dem_tr_pix_old
            )

            dem_line_tr_new.set_data(
                logt_new,
                dem_tr_pix_new
            )
        

        # ========================================================
        # UPDATE SELECTED POINT
        # ========================================================

        selected_point.set_data(
            [lrun[iy, ix]],
            [qrun[iy, ix]]
        )


        # ========================================================
        # UPDATE RIGHT PANEL
        # ========================================================

        ax_dem.set_title(
            f'iy={iy}, ix={ix}   '
            f'L={lrun[iy, ix]:.2e}, '
            f'Q={qrun[iy, ix]:.2e}'
        )

        ax_dem.relim()
        ax_dem.autoscale_view()
        ax_dem.set_xlim([5.0,8.0]) 

        # Prevent problematic log lower limit
        ymin, ymax = ax_dem.get_ylim()

        positive_values = np.concatenate([
            dem_cor_pix_old[dem_cor_pix_old > 0],
            dem_tr_pix_old[dem_tr_pix_old > 0],
            dem_cor_pix_new[dem_cor_pix_new > 0],
            dem_tr_pix_new[dem_tr_pix_new > 0]
        ])

        if len(positive_values) > 0:
            ymin = np.nanmin(positive_values) * 0.5
            ymax = np.nanmax(positive_values) * 2

            #ax_dem.set_ylim(ymin, ymax)


        fig.canvas.draw_idle()
        #plt.show()
        outfile = 'frame_Lind'+format('%0.4d'%l_ind)+'_Qind'+format('%0.4d'%q_ind)+'.png'
        plt.savefig(os.path.join('plots',outfile))
        




