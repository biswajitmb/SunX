pro run_llf2

FORWARD_FUNCTION AIA_FOV2Pix,XRT_FOV2Pix,HMI_FOV2Pix

;============ Inputs =============

data_dir = '/Users/bmondal/BM_Works/AR12749/0516UT/data/HMI/cutouts/'
HMI_files = file_search(data_dir+'HMIcut_lev1.5_region_01*newOBS.fits')

alpha = [0.01];,[-0.01 -0.005, 0 , 0.005, 0.01]

z_pix_hight = 500 ;in pixcel unit of HMI

OutPut_Dir = data_dir+'../IDL/'

;==================================

for i=0,n_elements(HMI_files)-1 do begin
    
    name = (strsplit((strsplit(HMI_files[i],'/',/extract))[-1],'.',/extract))[0]
    
    ;Plot HMI magnetogram
    WINDOW, 1, XSIZE=400, YSIZE=400
    read_sdo,HMI_files[i],ind,dao,/uncomp_delete
    
    bz = dao
    s = size(bz)

    LOADCT, 0
    plot_image,bz,XTICKLAYOUT=1,YTICKLAYOUT=1,XTICKFORMAT="(A1)",YTICKFORMAT="(A1)",POSITION=[0.0,0.0,1.0,1.0],MAX=100,MIN=-100
    ;save the HMI plot in .tiff file
    ;filename = OutPut_Dir+name+'.tiff'
    ;WRITE_TIFF, filename, ORIENTATION=0,TVRD(/TRUE)
    ;stop
    
    ;Extrapolate fields
    ;gx_bz2lff,bz,z_pix_hight,[1,1,1],bcube,alpha1=0.0
    drv = [0.5,0.5,0.5] ;voxel size in each directions
    ;N_x = 271
    for j=0,n_elements(alpha)-1 do begin
        gx_bz2lff,bz,z_pix_hight,drv,bcube,alpha1=alpha[j]
        
        bx = bcube[*,*,*,0]
        by = bcube[*,*,*,1]
        bz = bcube[*,*,*,2]
        
        save,bx,by,bz,filename= OutPut_Dir+name+'_Extrapolated_alp_'+string(alpha[j],format='(f0.4)')+'.sav'
    endfor
endfor
stop
end
