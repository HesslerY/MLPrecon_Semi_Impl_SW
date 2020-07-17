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
learning_set=[]
for i in range((base_day+0)*days_to_timesteps+1,(base_day+14)*days_to_timesteps,increment):
  learning_set.append(i)
for i in range((base_day+21)*days_to_timesteps,(base_day+21+14)*days_to_timesteps,increment):
  learning_set.append(i)
for i in range((base_day+42)*days_to_timesteps,(base_day+42+14)*days_to_timesteps,increment):
  learning_set.append(i)
for i in range((base_day+63)*days_to_timesteps,(base_day+63+14)*days_to_timesteps,increment):
  learning_set.append(i)
for i in range((base_day+84)*days_to_timesteps,(base_day+84+14)*days_to_timesteps,increment):
  learning_set.append(i)

for i in range((base_day+16*1)*days_to_timesteps,(base_day+16*1+4)*days_to_timesteps,increment):
  learning_set.append(i)
for i in range((base_day+16*2)*days_to_timesteps,(base_day+16*2+4)*days_to_timesteps,increment):
  learning_set.append(i)
for i in range((base_day+16*3)*days_to_timesteps,(base_day+16*3+4)*days_to_timesteps,increment):
  learning_set.append(i)
for i in range((base_day+16*4)*days_to_timesteps,(base_day+16*4+4)*days_to_timesteps,increment):
  learning_set.append(i)
for i in range((base_day+16*5)*days_to_timesteps,(base_day+16*5+4)*days_to_timesteps,increment):
  learning_set.append(i)

# 43200
def save_pair(x,y,fname):
    import h5py
    hf = h5py.File(fname, 'w')
    hf.create_dataset('x', data=x)
    hf.create_dataset('y', data=y)
    hf.close()
    return

#inputs=max(zonal_ext,merid_ext*2+1)*3+ ((merid_ext-1)*2+1)*2+ ((merid_ext-2)*2+1)*2  #zonal bands for 3 inputs+stencil
inputs=1+(zonal_ext*2+1)*7*(merid_ext*2+1)
outputs=zonal_ext_o*2+1 #1
print(inputs, outputs)

x=np.zeros(((len(learning_set))*grid_zonal,inputs))
y=np.zeros(((len(learning_set))*grid_zonal,outputs))

print ((len(learning_set))*grid_zonal)

latitudes=np.zeros(256)

file_grid = path+'/Precon7_H_exp3_time0_codes_FF_bits52.txt'
xcoord, ycoord=np.loadtxt( file_grid, usecols=(0,1), unpack=True)

ncols, nrows = len(set(xcoord)), len(set(ycoord)) 
latitudes= sorted(set(ycoord))


file_H0_in = path+'/Topo/'+'Timestep'+str(0)+'.txt'
H0_in=np.loadtxt( file_H0_in, usecols=(0), unpack=True)
grid_H0_in = np.flipud(H0_in.reshape((nrows, ncols), order='F'))

               
Mean_R_iter_in=0.0
Mean_a11_in   =0.0
Mean_a12_in   =0.0
Mean_a21_in   =0.0
Mean_a22_in   =0.0
Mean_b11_in   =0.0
Mean_b22_in   =0.0


for time in learning_set[0:2]: #range(start_learning,end_learning+1):      

    file_h_in = path+'/H_in/'+'Timestep'+str(time)+'.txt'
    file_h_out = path+'/H_in/'+'Timestep'+str(time+1)+'.txt'

    file_R_iter = path+'/R_iter/'+'Timestep'+str(time)+'_iter0.txt'
    file_a11 = path+'/a11/'+'Timestep'+str(time)+'.txt'
    file_a12 = path+'/a12/'+'Timestep'+str(time)+'.txt'
    file_a21 = path+'/a21/'+'Timestep'+str(time)+'.txt'
    file_a22 = path+'/a22/'+'Timestep'+str(time)+'.txt'
    file_b11 = path+'/b11/'+'Timestep'+str(time)+'.txt'
    file_b22 = path+'/b22/'+'Timestep'+str(time)+'.txt'

    h_in=np.loadtxt( file_h_in, usecols=(0), unpack=True)
    h_out=np.loadtxt( file_h_out, usecols=(0), unpack=True)

    R_iter_in=np.loadtxt( file_R_iter, usecols=(0), unpack=True)
    a11_in   =np.loadtxt( file_a11, usecols=(0), unpack=True)
    a12_in   =np.loadtxt( file_a12, usecols=(0), unpack=True)
    a21_in   =np.loadtxt( file_a21, usecols=(0), unpack=True)
    a22_in   =np.loadtxt( file_a22, usecols=(0), unpack=True)
    b11_in   =np.loadtxt( file_b11, usecols=(0), unpack=True)
    b22_in   =np.loadtxt( file_b22, usecols=(0), unpack=True)


    R_iter_in = R_iter_in /9.80616


    Min_R_iter_in =min(R_iter_in[:])       
    Min_a11_in    =min(a11_in[:])     
    Min_a12_in    =min(a12_in[:])     
    Min_a21_in    =min(a21_in[:])    
    Min_a22_in    =min(a22_in[:])     
    Min_b11_in    =min(b11_in[:])     
    Min_b22_in    =min(b22_in[:])     
    
    Max_R_iter_in =max(R_iter_in[:])                
    Max_a11_in    =max(a11_in[:])     
    Max_a12_in    =max(a12_in[:])     
    Max_a21_in    =max(a21_in[:])    
    Max_a22_in    =max(a22_in[:])     
    Max_b11_in    =max(b11_in[:])     
    Max_b22_in    =max(b22_in[:])     

print(Max_R_iter_in, Min_R_iter_in, Mean_R_iter_in) 

print(Max_a11_in, Min_a11_in, Mean_a11_in) 
print(Max_a12_in, Min_a12_in, Mean_a12_in) 

print(Max_a21_in, Min_a21_in, Mean_a21_in) 
print(Max_a22_in, Min_a22_in, Mean_a22_in) 

print(Max_b11_in, Min_b11_in, Mean_b11_in) 
print(Max_b22_in, Min_b22_in, Mean_b22_in) 

for time in learning_set: #range(start_learning,end_learning+1):      

    file_h_in = path+'/H_in/'+'Timestep'+str(time)+'.txt'
    file_h_out = path+'/H_in/'+'Timestep'+str(time+1)+'.txt'

    file_R_iter = path+'/R_iter/'+'Timestep'+str(time)+'_iter0.txt'
    file_a11 = path+'/a11/'+'Timestep'+str(time)+'.txt'
    file_a12 = path+'/a12/'+'Timestep'+str(time)+'.txt'
    file_a21 = path+'/a21/'+'Timestep'+str(time)+'.txt'
    file_a22 = path+'/a22/'+'Timestep'+str(time)+'.txt'
    file_b11 = path+'/b11/'+'Timestep'+str(time)+'.txt'
    file_b22 = path+'/b22/'+'Timestep'+str(time)+'.txt'

    h_in=np.loadtxt( file_h_in, usecols=(0), unpack=True)
    h_out=np.loadtxt( file_h_out, usecols=(0), unpack=True)

    R_iter_in=np.loadtxt( file_R_iter, usecols=(0), unpack=True)
    a11_in   =np.loadtxt( file_a11, usecols=(0), unpack=True)
    a12_in   =np.loadtxt( file_a12, usecols=(0), unpack=True)
    a21_in   =np.loadtxt( file_a21, usecols=(0), unpack=True)
    a22_in   =np.loadtxt( file_a22, usecols=(0), unpack=True)
    b11_in   =np.loadtxt( file_b11, usecols=(0), unpack=True)
    b22_in   =np.loadtxt( file_b22, usecols=(0), unpack=True)


    R_iter_in = R_iter_in /9.80616


    Min_R_iter_in =min(Min_R_iter_in,min(R_iter_in[:]))       
    Min_a11_in    =min(Min_a11_in,min(a11_in[:]))     
    Min_a12_in    =min(Min_a12_in,min(a12_in[:]))     
    Min_a21_in    =min(Min_a21_in,min(a21_in[:]))   
    Min_a22_in    =min(Min_a22_in,min(a22_in[:]))     
    Min_b11_in    =min(Min_b11_in,min(b11_in[:]))     
    Min_b22_in    =min(Min_b22_in,min(b22_in[:]))     
    
    Max_R_iter_in =max(Max_R_iter_in,max(R_iter_in[:]))                
    Max_a11_in    =max(Max_a11_in,max(a11_in[:]))     
    Max_a12_in    =max(Max_a12_in,max(a12_in[:]))     
    Max_a21_in    =max(Max_a21_in,max(a21_in[:]))    
    Max_a22_in    =max(Max_a22_in,max(a22_in[:]))     
    Max_b11_in    =max(Max_b11_in,max(b11_in[:]))     
    Max_b22_in    =max(Max_b22_in,max(b22_in[:]))     

    Mean_R_iter_in=Mean_R_iter_in+np.mean(R_iter_in[:])/float(len(learning_set))
    Mean_a11_in   =Mean_a11_in+np.mean(a11_in[:])/float(len(learning_set))     
    Mean_a12_in   =Mean_a12_in+np.mean(a12_in[:])/float(len(learning_set))     
    Mean_a21_in   =Mean_a21_in+np.mean(a21_in[:])/float(len(learning_set))     
    Mean_a22_in   =Mean_a22_in+np.mean(a22_in[:])/float(len(learning_set))     
    Mean_b11_in   =Mean_b11_in+np.mean(b11_in[:])/float(len(learning_set))     
    Mean_b22_in   =Mean_b22_in+np.mean(b22_in[:])/float(len(learning_set))     



print(Max_R_iter_in, Min_R_iter_in, Mean_R_iter_in) 

print(Max_a11_in, Min_a11_in, Mean_a11_in) 
print(Max_a12_in, Min_a12_in, Mean_a12_in) 

print(Max_a21_in, Min_a21_in, Mean_a21_in) 
print(Max_a22_in, Min_a22_in, Mean_a22_in) 

print(Max_b11_in, Min_b11_in, Mean_b11_in) 
print(Max_b22_in, Min_b22_in, Mean_b22_in) 





























































