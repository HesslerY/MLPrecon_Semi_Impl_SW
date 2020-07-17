import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

path='../data/data_ADI_NOTgcr_EXP3_Dp_1M10_res4'

grid_zonal=512
grid_merid=256
dt=200

zonal_ext=2
zonal_ext_o=0
merid_ext=2

start_learning= 5000 # 5000
end_learning=  22000 #10000

start_valid= 22001
end_valid=  27000

base_day=0
days_to_timesteps=(24*3600)/dt
increment=8


validation_set=[]
for i in range((base_day+16*1)*days_to_timesteps,(base_day+16*1+4)*days_to_timesteps,increment):
  validation_set.append(i)
for i in range((base_day+16*2)*days_to_timesteps,(base_day+16*2+4)*days_to_timesteps,increment):
  validation_set.append(i)
for i in range((base_day+16*3)*days_to_timesteps,(base_day+16*3+4)*days_to_timesteps,increment):
  validation_set.append(i)
for i in range((base_day+16*4)*days_to_timesteps,(base_day+16*4+4)*days_to_timesteps,increment):
  validation_set.append(i)
for i in range((base_day+16*5)*days_to_timesteps,(base_day+16*5+4)*days_to_timesteps,increment):
  validation_set.append(i)

iteration=1

#inputs=max(zonal_ext,merid_ext*2+1)*3+ ((merid_ext-1)*2+1)*2+ ((merid_ext-2)*2+1)*2  #zonal bands for 3 inputs+stencil

latitudes=np.zeros(256)



file_grid = path+'/Precon7_H_exp3_time0_codes_FF_bits52.txt'
xcoord, ycoord=np.loadtxt( file_grid, usecols=(0,1), unpack=True)

ncols, nrows = len(set(xcoord)), len(set(ycoord)) 
latitudes= sorted(set(ycoord))


file_H0_in = path+'/Topo/'+'Timestep'+str(0)+'.txt'
H0_in=np.loadtxt( file_H0_in, usecols=(0), unpack=True)
grid_H0_in = np.flipud(H0_in.reshape((nrows, ncols), order='F'))

orig_error=np.zeros(((len(validation_set))*grid_zonal))
iter_error=np.zeros(((len(validation_set))*grid_zonal))




for lat in range(0,nrows):
  print(lat)
  orig_error[:]=0.0
  iter_error[:]=0.0
  timecounter=0
  for time in validation_set:

    file_h_in = path+'/H_in/'+'Timestep'+str(time)+'.txt'
    file_R_in = path+'/RHS/'+'Timestep'+str(time)+'.txt'
    file_h_out = path+'/H_in/'+'Timestep'+str(time+1)+'.txt'
    file_h_iter = path+'/H_iter/'+'Timestep'+str(time)+'_iter'+str(iteration)+'.txt'

    h_in=np.loadtxt( file_h_in, usecols=(0), unpack=True)
    R_in=np.loadtxt( file_R_in, usecols=(0), unpack=True)
    h_out=np.loadtxt( file_h_out, usecols=(0), unpack=True)
    h_iter=np.loadtxt( file_h_iter, usecols=(0), unpack=True)

    grid_h_in = np.flipud(h_in.reshape((nrows, ncols), order='F'))
    grid_R_in = np.flipud(R_in.reshape((nrows, ncols), order='F'))
    grid_h_iter = np.flipud(h_iter.reshape((nrows, ncols), order='F'))

    grid_h_out = np.flipud(h_out.reshape((nrows, ncols), order='F'))

    grid_h_in = grid_h_in /9.80616

    grid_h_out=grid_h_out /9.80616
    grid_R_in = grid_R_in /9.80616

    grid_h_iter = grid_h_iter /9.80616
    #print(time)
    #print(grid_h_iter)
    #print(grid_h_in)
    #print(grid_h_out)

    grid_h_in= grid_h_in-grid_h_out   # I want to predict the error
    grid_h_iter= grid_h_iter-grid_h_out

    grid_R_in= grid_h_in-grid_R_in 
  
    for lon in range(0,ncols):
      orig_error[(timecounter)*grid_zonal+lon]=grid_h_in[lat,lon]
      iter_error[(timecounter)*grid_zonal+lon]=grid_h_iter[lat,lon]

    timecounter+=1

  print(str(lat))
  print(np.mean(np.absolute(orig_error)) )
  print(np.mean(np.absolute(iter_error)) )
  f = open("truth_diff_METRICnorm_iter_1_Lats.txt", "a")
  f.write(str(lat)+' '+str(np.mean(np.absolute(orig_error))) +' '+ str(np.mean(np.absolute(orig_error)))  +' '+str(np.mean(np.absolute(iter_error)) ) + '\n')
  f.close()


  
