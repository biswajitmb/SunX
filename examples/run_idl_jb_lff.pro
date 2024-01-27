
FORWARD_FUNCTION j_b_lff,gx_bz2lff

PRO run_idl_jb_lff,x=x,y=y,bz0=bz0
    x =  [FINDGEN(50)/4.95 - 10 , FINDGEN(50)/4.95 + 0.101009] # REPLICATE(1,100)
    y = TRANSPOSE(x)
    bz0 = 50 * (EXP(-0.1 * (x - 5)^2 - 0.1 * (y - 5)^2) - EXP(-0.1 * (x + 5)^2 - 0.1 * (y + 5)^2))
    ;surface,bz0,x,y
    ;plot_image,bz0 ;ssw require 
    gx_bz2lff,bz0,50,[0.5,0.5,0.5],bcube,alpha1=0.0
    save,bcube,x,y,filename='test_idl_out.sav'
    ;surface,bcube[*,*,10,0] 
END

;+
;
; NAME :
;          J_B_LFF
; PURPOSE :
;          Linear Force-Free Extrapolation of Magnetic Field
; CALLING SEQUENCE:
;          J_B_LFF, bz0, z, bx, by, bz, alpha=alpha, seehafer=seehafer
;
; INPUT :
;           bz0 : 2-d array of vertical field at z=0 plane
;           z   : 1-d array of heights (in unit of pixels)
; OUTPUT :
;           bx, by, bz : 3-d arrays of field components
; MODIFICATION HISTORY:
;                    J.E.R.Costa 2004, v.1
;                    Reference: Nakagawa and Raadu 1972, Solar Physics, 25, 127
;                               Seehafer 1978, Solar Physics, 58, 215
;                    jercosta: alpha1 is an input keyword
;-
pro j_b_lff,  bz0, z, bx, by, bz, alpha1=alpha1,seehafer=seehafer $
           , sub_b00=sub_b00, sub_plan=sub_plan

if n_elements(alpha1) eq 0 then alpha1=0.

nx1=n_elements(bz0(*,0))
ny1=n_elements(bz0(0,*))
nz=n_elements(z)
b00=0.0
if (keyword_set(sub_b00) and keyword_set(sub_plan)) then begin
  print, 'You cannot set both keyword: sub_b00 and sub_plan together!!'
  return
endif
if keyword_set(sub_b00) then b00=mean(bz0)
if keyword_set(sub_plan) then b00=sfit(bz0,1)


if keyword_set(seehafer) then begin
    nx=2*nx1  & ny=2*ny1
    bz0e=fltarr(nx, ny)
    bz0e(0:nx1-1, 0:ny1-1)=bz0-b00
    bz0e(nx1:nx-1, 0:ny1-1)= - rotate(bz0, 5)
    bz0e(0:nx1-1, ny1:ny-1)= - rotate(bz0, 7)
    bz0e(nx1:nx-1, ny1:ny-1)= -rotate(bz0e(0:nx1-1, ny1:ny-1), 5)
endif else begin
    nx=nx1 & ny= ny1
    bz0e=bz0-b00
endelse
kx = 2*!pi*[findgen(nx/2+1),reverse(-1-findgen(nx-nx/2-1))]/nx
ky = 2*!pi*[findgen(ny/2+1),reverse(-1-findgen(ny-ny/2-1))]/ny

if abs(alpha1) ge 1. then begin
    print, 'The magnitude of alpha is too big! '
    print, '|alpha| should be less than 1.'
    return
endif
alpha=alpha1
print,'alpha=',alpha
kx=kx#replicate(1, ny)
ky=replicate(1,nx)#ky
fbz0 = fft(bz0e, -1)
kz=sqrt((kx^2+ky^2 - alpha^2)>0.)  ; Positive solutions  
                                   ; see Nakagawa e Raadu p. 132
argx=fbz0*complex(0,-1)*(kx*kz-alpha*ky)/((kz^2+ alpha^2)>kx(1,0)^2)
argy=fbz0*complex(0,-1)*(ky*kz+alpha*kx)/((kz^2+alpha^2)>ky(0,1)^2)
bx=fltarr(nx1,ny1,nz)
by=bx
bz=bx
for j=0, nz-1 do begin
    bx(*,*,j) = (float(fft(argx*exp(-kz*z(j)),1)))[0:nx1-1, 0:ny1-1]
    by(*,*,j) = (float(fft(argy*exp(-kz*z(j)),1)))[0:nx1-1, 0:ny1-1]
    bz(*,*,j) = (float(fft(fbz0*exp(-kz*z(j)),1)))[0:nx1-1, 0:ny1-1]+b00
endfor
help, kx
end

pro gx_bz2lff,Bz,nz,dr,Bcube,alpha1=alpha1,seehafer=seehafer, sub_b00=sub_b00, sub_plan=sub_plan
  
  ;bz must be a two dimensional array containing the base Bz data
  sz=size(bz)
  Nx=sz[1]
  Ny=sz[2]
  N=[Nx,Ny]
  if sz[0] ge 4 then bz=reform(bz[*,*,*,2])
  if sz[0] ge 3 then bz=reform(bz[*,*,sz[3]-1])

  ;Nz specifies the height of the extrapolation
  if n_elements(Nz) eq 0 then Nz= min([Nx,Ny])

  N=long([N,Nz])

  ;dr represents voxel size in each direction (arcsec)
  if n_elements(dr) eq 0 then dr=[2.,2.,2.]
  z=findgen(Nz)*dr[2]/dr[0]
  ;use J.R.Costa j_b_lff extrapolation routine
  j_b_lff,  bz, z, bxout, byout, bzout,alpha1=alpha1,seehafer=seehafer, sub_b00=sub_b00, sub_plan=sub_plan
  ;Bcube is the four dimensional datacube output
 ;Bcube is the four dimensional datacube output
  Bcube=fltarr(Nx,Ny,Nz,3)
  Bcube[*,*,*,0]=temporary(bxout)
  Bcube[*,*,*,1]=temporary(byout)
  Bcube[*,*,*,2]=temporary(bzout)
end


