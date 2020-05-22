import os
from ctypes import *
import time
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
def sin_func(x,a,b,c,d):
    return a * np.sin(b * (x - c)) + d 
 
os.chdir('C:/Users/saurs/OneDrive/Desktop/EM/20190917/')#import 장소.
from dwfconstants import *
if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")
#dll 불러오기
WaveNum=0
OscilNum=2
WaveChannel = c_int(WaveNum)
OscilChannel= c_int(OscilNum)

nSamples = 700000
hzAcq= c_double(700000)# Acquired data per sec
RTime = nSamples/hzAcq.value
hdwf = c_int() #handle 
sts= c_byte() #AnalogIn status info
cAvailable = c_int()
cBufMax=c_int()
cLost = c_int()
cCorrupted = c_int()
fLost =0 
fCorrupted = 0

Hz=35
Hz_i=2
k=0
Amp=3



os.chdir('C:/Users/saurs/OneDrive/Desktop/EM/Module2')
########################################################
while Hz<=0.5*hzAcq.value:
    params1,params2=0,0
    Phase=0
    err=0
    rgdSamples=(c_double*nSamples)() # Os1
    rgdSamples1=(c_double*nSamples)()  #Os2

    if k==3:
        print('앜ㅋㅋㅋ')
        break;
    wavelength=nSamples/(RTime*Hz)
    dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))
    if hdwf.value == hdwfNone.value:
        print("failed to open device")
        quit;   

    dwf.FDwfAnalogOutNodeEnableSet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_bool(True))
    dwf.FDwfAnalogOutNodeFunctionSet(hdwf, WaveChannel, AnalogOutNodeCarrier, funcSine)
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Hz))
    dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Amp))
    dwf.FDwfAnalogOutNodeOffsetSet(hdwf,WaveChannel,AnalogOutNodeCarrier, c_double(0))

    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_bool(True))
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(1), c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(1), c_double(15))
    dwf.FDwfAnalogInChannelRangeSet(hdwf,c_int(0), c_double(15))
    dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
    dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
    dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(RTime))
    #a=input('If the thing is connected, press any button')
##########################################################
    dwf.FDwfAnalogOutConfigure(hdwf,WaveChannel,c_bool(True))
    time.sleep(0.1)
    dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_bool(True))
#    time.sleep(0.5)
    cSamples = 0
    

    while cSamples < nSamples:
        dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
#        if cSamples%100 ==0:
#            print(datetime.now())
        if cSamples == 0 and (sts == DwfStateConfig or sts == DwfStatePrefill or sts == DwfStateArmed) :
            # Acquisition not yet started.
            continue
    
        dwf.FDwfAnalogInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))
        cSamples += cLost.value
    
        if cLost.value :
            fLost = 1
        if cCorrupted.value :
            fCorrupted = 1
    
        if cAvailable.value==0 :
            break;
    
        if cSamples+cAvailable.value > nSamples :
            cAvailable = c_int(nSamples-cSamples)

        dwf.FDwfAnalogInStatusData(hdwf, c_int(0), byref(rgdSamples, sizeof(c_double)*cSamples), cAvailable) # get channel 1 data
        dwf.FDwfAnalogInStatusData(hdwf, c_int(1), byref(rgdSamples1, sizeof(c_double)*cSamples), cAvailable) # get channel 1 data

        cSamples += cAvailable.value
    dwf.FDwfAnalogOutReset(hdwf, WaveChannel)
    dwf.FDwfDeviceCloseAll()    
    if fLost:
        print("Samples were lost! Reduce frequency")
        break;
    if fCorrupted:
        print("Samples could be corrupted! Reduce frequency")
        break;
    wavelength=int(nSamples/Hz)
    x_data=np.arange(0,nSamples,1)
    #x=np.arange(100,1000,0.01)
    params1,pcov1= optimize.curve_fit(sin_func, x_data[20:Hz_i*wavelength], rgdSamples[20:Hz_i*wavelength],p0=[Amp,2*np.pi/wavelength,0.5*wavelength,1])
    params2,pcov2= optimize.curve_fit(sin_func, x_data[20:Hz_i*wavelength], rgdSamples1[20:Hz_i*wavelength],p0=[Amp,2*np.pi/wavelength, 0.5*wavelength,1])
#    params1,pcov1= optimize.curve_fit(sin_func, x_data, rgdSamples,p0=[Amp,2*np.pi/wavelength,0,0],bounds=([0,-np.inf,0,-np.inf],np.inf))
#    params2,pcov2= optimize.curve_fit(sin_func, x_data, rgdSamples1,p0=[Amp,2*np.pi/wavelength,0,0],bounds=([0,-np.inf,0,-np.inf],np.inf))

    
    plt.figure(figsize=(20,15))
    #y1,y2= sin_func(x, params1[0],params1[1],params1[2]), sin_func(x_data,params2[0],params2[1],params2[2])
    plt.plot(x_data[wavelength:5*wavelength], sin_func(x_data[wavelength:5*wavelength], *params1),'y*',label='fit 1',markersize=4)
    plt.plot(x_data[wavelength:5*wavelength], sin_func(x_data[wavelength:5*wavelength], *params2),'yo',label='fit 2',markersize=4)
    plt.plot(x_data[wavelength:5*wavelength],rgdSamples[wavelength:5*wavelength],'ro',markersize=4,label='RealData1')
    plt.plot(x_data[wavelength:5*wavelength],rgdSamples1[wavelength:5*wavelength],'go',markersize=4,label='RealData2')
    plt.legend()
    plt.show()
    perr1 = np.sqrt(np.diag(pcov1))
    perr2 = np.sqrt(np.diag(pcov2))
    
    err1=1e-3
    if params1[1] - params2[1] >err1 :
        print('not reliable fitting, Frequency')
        break;
        '''
    if params1[0]*params2[0]>0:
        Phase = (params1[2]-params2[2])
    elif params1[0]<0 and params2[0]>0:
        Phase = (-params1[2]+params2[2])
        
    if params1[0]*params2[0] > 0:
#        print(360*(params1[2] -params2[2])/(2*np.pi))
        Phase=params1[2]-params2[2]
    elif params1[0]<0 and params2[0]>0:
#        print(360*(np.pi-params1[2] -params2[2])/(2*np.pi))
        Phase=np.pi-params1[2]-params2[2]
    elif params1[0]>0 and params2[0]<0:
#        print(360*(params1[2] + params2[2]-np.pi)/(2*np.pi))
        Phase=params1[2]+params2[2] -np.pi
    while np.abs(Phase)< 0.5*np.pi:
        if Phase > 0.5* np.pi:
            Phase=2*np.pi - Phase
            continue
        elif Phase < -0.5 * np.pi:
            Phase = 2*np.pi- Phase
            continue
        else:
            break
    '''
#    err = (np.abs(params1[2]*perr1[2]) +np.abs(params2[2]* perr2[2]))/wavelength
    err=1e-2
    print(params1[2],params2[2])
    if params1[0]<0 and params2[0]>0 :
        if params1[2] < 0:
            params1[2] += 0.5*wavelength
        else:
            params1[2] -=0.5*wavelength
    elif params2[0]<0 and params1[0]>0:
        if params2[2] <0:
            params2[2]  += 0.5*wavelength
        else:
            params2[2] -= 0.5*wavelength
    print(params1[2],params2[2])
    Phase = (params1[2]-params2[2])/wavelength
    while np.abs(Phase)>np.pi:
        Phase= 2*np.pi - np.abs(Phase)
    if np.abs(Amp-np.abs(params1[0]))> 0.5: print('fitting error,Amplitude');break;
    elif -err<=Phase<=err and k==2:
        params1,params2=np.abs(params1),np.abs(params2)
        Resist = (np.abs(params2[0]))/(np.abs(params1[0])-np.abs(params2[0])) * 220 
        Rerr= (-1e6*Resist/(1e6+Resist) + Resist) + 220e-2
        print('Resistance {:.4e} Ohm , Error is {:.4e} '.format(Resist, Rerr))
        break;
    elif Phase<-err :
        Capacitance = -1/(np.tan(Phase)*440*np.pi*Hz)
        Cerr = 1/(np.tan(np.sqrt(perr1[2]**2+perr2[2]**2))*220*2*np.pi*Hz)
        print('Capacitor {:.4e} F, error is {:.4e} F'.format(Capacitance,Cerr))
        break;
    elif Phase>err:
        Henry = np.tan(Phase)*220/(2*np.pi*Hz)
        print('Coil %.4e H' %Henry)        
        break;
    elif np.abs(Phase) > 0.5*np.pi:
        print('unreliable fitting')
        break;

    Hz = 20*Hz        
    Hz_i +=50
    k+=1
    
