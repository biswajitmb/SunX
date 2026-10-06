pro make_idl_savfile

;file = './outputs/ebtel_runs.h5'
file = '/Volumes/Working/ebtel/outputs/ebtel_runs_final.h5'

fid = H5F_OPEN(file)

LOGTDEM = H5D_READ(H5D_OPEN(fid, 'LOGTDEM'))

LRUN = H5D_READ(H5D_OPEN(fid, 'LRUN'))
QRUN = H5D_READ(H5D_OPEN(fid, 'QRUN'))
TRUN = H5D_READ(H5D_OPEN(fid, 'TRUN'))

DEM_COR_RUN = H5D_READ(H5D_OPEN(fid, 'DEM_COR_RUN'))
DEM_TR_RUN = H5D_READ(H5D_OPEN(fid, 'DEM_TR_RUN'))

DDM_COR_RUN = H5D_READ(H5D_OPEN(fid, 'DDM_COR_RUN'))
DDM_TR_RUN = H5D_READ(H5D_OPEN(fid, 'DDM_TR_RUN'))

H5F_CLOSE, fid


HELP, LOGTDEM
HELP, LRUN
HELP, DEM_COR_RUN

SAVE, $
    LOGTDEM, $
    LRUN, $
    QRUN, $
    TRUN, $
    DEM_COR_RUN, $
    DEM_TR_RUN, $
    DDM_COR_RUN, $
    DDM_TR_RUN, $
    FILENAME='./outputs/ebtel_runs.sav'


stop
end
