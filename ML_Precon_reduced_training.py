import numpy as np
import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import rmsprop,adam
import keras.backend as K

def main(activation = 'relu',learning_rate=10.0**(-5),hidden_layers=5,width=200,Lat=9):
    #K.set_floatx('float64')
    inputs= 176 
    outputs= 1 
    print('load Training data')
    print(Lat)
    print(hidden_layers)
    print(width)
    #x, y = load_pair('../data/ML_data_ADI_NOTgcr_EXP3_Dp_res05/training32x2_32x0.h5')
    x, y = load_pair('../data/ML_data_ADI_NOTgcr_EXP3_Dp_1M10_res4/training_Lcoeff_R_2x2_1x0'+str(Lat)+'DP.h5')
    print(np.sqrt(np.mean(np.square(y))) )
    print(np.mean(np.absolute(y)) )
    print (x[0,:])
    print (y[0,:])
    print (x[-1,:])
    print (y[-1,:])

    print('load validation data')
    #x_val, y_val =  load_pair('../data/ML_data_ADI_NOTgcr_EXP3_Dp_res05/validation32x2_32x0.h5')
    x_val, y_val =  load_pair('../data/ML_data_ADI_NOTgcr_EXP3_Dp_1M10_res4/validation_Lcoeff_R_2x2_1x0'+str(Lat)+'DP.h5')
    print(np.sqrt(np.mean(np.square(y_val))) )
    print(np.mean(np.absolute(y_val)) )

    Maxima=np.zeros((inputs)) 
    Minima=np.zeros((inputs))
    Mean=np.zeros((inputs))
    MaximaO=np.zeros((outputs)) 
    MinimaO=np.zeros((outputs))
    MeanO=np.zeros((outputs))

    for k in range(0, len(y)):
     counter=0
     for i in range(1, inputs,7):
      counter+=1
      if (counter == 3 or counter == 11 or counter == 13 or counter == 15 or counter == 23):
        x[k,i]=x[k,i]
        #print(x[k,i])
      else:
        x[k,i]=0.0
        #print(x[k,i])
     #raw_input("Press Enter to continue...")

    for k in range(0, len(y_val)):
     counter=0
     for i in range(1, inputs,7):
      counter+=1
      if (counter == 3 or counter == 11 or counter == 13 or counter == 15 or counter == 23):
        x_val[k,i]=x_val[k,i]
      else:
        x_val[k,i]=0.0



    print(len(y))
    Scaling=np.zeros((len(y))) 
    print(len(y_val))
    ScalingO=np.zeros((len(y_val))) 

    for k in range(0, len(y)):
     for i in range(1, inputs,7):
      if( abs(x[k,i]) > Scaling[k] ):
       Scaling[k]    =abs(x[k,i])

    for k in range(0, len(y_val)):
     for i in range(1, inputs,7):
      if( abs(x_val[k,i]) > ScalingO[k] ):
       ScalingO[k]    =abs(x_val[k,i])


    Maxima[0]=3.5/2.0 #max(x[:,0])
    Minima[0]=-3.5/2.0 #min(x[:,0])
    Mean[0]=0.0

    Maxima[1]=1100.0 #max(x[:,i])
    Minima[1]=-1100.0 #min(x[:,i])  
    Mean[1]=0.0

    Maxima[2]=370735007156489.9 #21.0 #max(x[:,i])
    Minima[2]=   285797600451.5306
    Mean[2]  =  8832816614973.127 #0.0

    Maxima[3]=40927556954.599144 #21.0 #max(x[:,i])
    Minima[3]=-40927556954.599144 #-21.0 #min(x[:,i])  
    Mean[3]= 0.0     

    Maxima[4]=2814357893215.7007 #1.0 #max(x[:,i])
    Minima[4]=   5861279991.116582
    Mean[4]  =1632540745224.2573

    Maxima[5]= 40927556954.59915
    Minima[5]=-40927556954.59915
    Mean[5]=0.0

    Maxima[6]=12286263193393.56 #21.0 #max(x[:,i])
    Minima[6]=-12286263193393.56 #-21.0 #min(x[:,i])  
    Mean[6]=0.0

    Maxima[7]= 163170483547.3604 #21.0 #max(x[:,i])
    Minima[7]=-163170483547.3604 #-21.0 #min(x[:,i])  
    Mean[7]=0.0


    print('Maxima')
    print (Maxima)
    print('Minima')
    print (Minima)
    print('Mean')
    print (Mean)    

    print ('before')
    print(min(y), max(y) )
    print(min(y_val), max(y_val) )
    print(np.mean(y) )
    print(np.mean(y_val) )

    x[:,0]    =0.0 #(x[:,0]     )/(Maxima[0])
    x_val[:,0]=0.0 #(x_val[:,0] )/(Maxima[0])

    for k in range(0, len(y)):    
     for i in range(1, inputs,7):
      x[k,i]    = x[k,i] /Scaling[k]

    for k in range(0, len(y_val)):    
     for i in range(1, inputs,7):
      x_val[k,i]= x_val[k,i] /ScalingO[k]

    for i in range(2, inputs,7):
      x[:,i]    =0.0#(x[:,i]     -Mean[2])/(Maxima[2]-Minima[2])
      x_val[:,i]=0.0#(x_val[:,i] -Mean[2])/(Maxima[2]-Minima[2])
    for i in range(3, inputs,7):
      x[:,i]    =0.0#(x[:,i]     -Mean[3])/(Maxima[3]-Minima[3])
      x_val[:,i]=0.0#(x_val[:,i] -Mean[3])/(Maxima[3]-Minima[3])
    for i in range(4, inputs,7):
      x[:,i]    =0.0#(x[:,i]     -Mean[4])/(Maxima[4]-Minima[4])
      x_val[:,i]=0.0#(x_val[:,i] -Mean[4])/(Maxima[4]-Minima[4])
    for i in range(5, inputs,7):
      x[:,i]    =0.0#(x[:,i]     -Mean[5])/(Maxima[5]-Minima[5])
      x_val[:,i]=0.0#(x_val[:,i] -Mean[5])/(Maxima[5]-Minima[5])
    for i in range(6, inputs,7):
      x[:,i]    =0.0#(x[:,i]     -Mean[6])/(Maxima[6]-Minima[6])
      x_val[:,i]=0.0#(x_val[:,i] -Mean[6])/(Maxima[6]-Minima[6])
    for i in range(7, inputs,7):
      x[:,i]    =0.0#(x[:,i]     -Mean[7])/(Maxima[7]-Minima[7])
      x_val[:,i]=0.0#(x_val[:,i] -Mean[7])/(Maxima[7]-Minima[7])



    for k in range(0, len(y)):    
     y[k]    = y[k] /Scaling[k]

    for k in range(0, len(y_val)):    
     y_val[k]= y_val[k] /ScalingO[k]


    #for i in range(0, inputs):
    #   print(min(x[:,i]), max(x[:,i]))
    #raw_input("Press Enter to continue...")

    print(min(y), max(y) )
    print(min(y_val), max(y_val) )
    print('Mean of Testoutput and Validationoutput')
    print(np.mean(y) )
    print(np.mean(y_val) )
    print(np.mean(Scaling) )
    print(np.mean(ScalingO) )
    print(min(Scaling), max(Scaling) )
    print(min(ScalingO), max(ScalingO) )

    print ('normalised')
    print (x[0,:])
    print (y[0,:])
    print(Scaling[0])
    print (x_val[0,:])
    print (y_val[0,:])
    print(ScalingO[0])
    assert x.shape[1] == inputs
    assert y.shape[1] == outputs
    assert x_val.shape[1] == inputs
    assert y_val.shape[1] == outputs

    #Create model
    model = Sequential()
    #Add first layer (which has a different input shape to match your data)
    if (width==1 and hidden_layers==0 ):
      print('Linear Model')

      model.add(Dense(width,input_shape=(inputs,),activation='linear'))

    else:
      model.add(Dense(width,input_shape=(inputs,),activation=activation))
      #Add the rest of the hidden layers
      for i in range(1,hidden_layers):
        print('add layer'+str(i))
        model.add(Dense(width,activation=activation))
      #Add the final layer (I recommend linear)
      model.add(Dense(outputs,activation='linear'))
    #Make an optimiser with a learning rate
    opt = adam(lr=learning_rate)
    #Compile the model
    model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])

    #This step is extra complexity. It tells Keras save the best model (overwriting the old one)
    # and also to quit if it stops improving the error on the validation data
    #You have to load the best model back as this won't necessarily be the last model.
    callbacks=[EarlyStopping(monitor='val_loss', patience=10), \
                ModelCheckpoint(filepath='ModelWeights/model2x2_1x0_l'+str(hidden_layers)+'w'+str(width)+'_METRICnorm_Lat_Lcoeff_R_uselinear_noinputact_reduced_r_only_r'+str(Lat)+'_ext_DP.h5', \
                monitor='val_loss', save_best_only=True)]
    #Train the model
    model.fit(x,y,validation_data=(x_val,y_val),callbacks=callbacks,epochs=50)

    #Load the best model
    model = load_model('ModelWeights/model2x2_1x0_l'+str(hidden_layers)+'w'+str(width)+'_METRICnorm_Lat_Lcoeff_R_uselinear_noinputact_reduced_r_only_r'+str(Lat)+'_ext_DP.h5')
    #Check it gets the same answer on the validation data
    error=model.evaluate(x_val,y_val)
    y_ML_model = model.predict(x_val)

    for k in range(0, len(y_val)): 
     y_ML_model[k]=y_ML_model[k]*ScalingO[k]
     y_val[k]=y_val[k]*ScalingO[k]

    print(y_ML_model.shape[0])
    print(np.sqrt(np.mean(np.square(y_ML_model))) )
    print ('MAE of y_ML_model and y')
    print(np.mean(np.absolute(y_ML_model)) )
    print(np.mean(np.absolute(y)) )
    print(np.sqrt(np.mean(np.square(y_val-y_ML_model))) )
    print(np.mean(np.absolute(y_val-y_ML_model)) )
    print(error)

    print(np.mean(np.absolute( (y_val[0:5]-y_ML_model[0:5]) )))

    print(np.mean(np.absolute( (y_val[0:1800*64]-y_ML_model[0:1800*64]) )))
    print(np.mean(np.absolute( (y_val[1*1800*64 :2*1800*64]-y_ML_model[1*1800*64 :2*1800*64])  )))
    print(np.mean(np.absolute( (y_val[2*1800*64 :3*1800*64]-y_ML_model[2*1800*64 :3*1800*64])  )))
    print(np.mean(np.absolute( (y_val[3*1800*64 :4*1800*64]-y_ML_model[3*1800*64 :4*1800*64])  )))
    print(np.mean(np.absolute( (y_val[4*1800*64 :]-y_ML_model[4*1800*64 :])  )))


    f = open('model2x2_1x0_l'+str(hidden_layers)+'w'+str(width)+'_METRICnorm_Lcoeff_R_uselinear_Lats_ext_DP_noinputact_reduced_r_only_r.txt', "a")
    f.write(str(Lat)+' '+str(np.mean(np.absolute(y_val)) )  +'  MAE_Error '+str(np.mean(np.absolute(y_ML_model-y_val)) ) + '  Model_MA '+ str(np.mean(np.absolute(y_ML_model)) )+ '  True_MA '+ str(np.mean(np.absolute(y)) ) +'  Max_delta_Phi '+ str(np.amax(np.absolute(y_val)) )+'  Max_Error '+ str(np.amax(np.absolute(y_val-y_ML_model)) ) +'\n')
    f.close()
    #for i in range(0,700):
      #print(i)
      #print(y_val[i]*(MaximaO[0]-MinimaO[0])+MeanO[0] , y_ML_model[i]*(MaximaO[0]-MinimaO[0])+MeanO[0], y_val[i]*(MaximaO[0]-MinimaO[0])+MeanO[0] - (y_ML_model[i]*(MaximaO[0]-MinimaO[0])+MeanO[0]))
      #print(y_val[i], y_ML_model[i], y_val[i] - y_ML_model[i])
      #raw_input("Press Enter to continue...")

def save_pair(x,y,fname):
    import h5py
    hf = h5py.File(fname, 'w')
    hf.create_dataset('x', data=x)
    hf.create_dataset('y', data=y)
    hf.close()
    return

def load_pair(fname):
    import h5py
    hf = h5py.File(fname, 'r')
    x=hf['x'].value
    y=hf['y'].value
    hf.close()
    return x,y

for Latitude in range(0,256):
  main(width=1,hidden_layers=0,Lat=Latitude)   ## how good is the linear model?

#for w_width in [ 5]:
#    for Latitude in range(0,32):
#      main(width=w_width,hidden_layers=1,Lat=Latitude)

#for w_width in [200]:
#    for Latitude in range(31,32):
#      main(width=w_width,hidden_layers=5,Lat=Latitude)


