import os

def write_txt(list,fname,sep):

    file=open(fname,'w')
    vstr=''
    
    for a in list:
        vstr = vstr + str(a) +sep 
    vstr = vstr.rstrip(sep)
    
    file.writelines(vstr)
    file.close()
    print('[파일 저장 완료]')
#Ref: https://data-make.tistory.com/109

#%%
from ctypes import *
import time
import sys
from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os 
os.chdir('C:/Users/ZEST/Desktop/data')#import 장소.
from dwfconstants import *


if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")
#dll 불러오기
WaveNum=int(input('Channel number?: W1=0 , W2=1 \n'))
OscilNum=int(input('Oscilloscope number?: T1=0, T2=1, both=2, Reset=3 \n'))
WaveChannel = c_int(WaveNum)
OscilChannel= c_int(OscilNum)

nSamples = 10000
rgdSamples=(c_double*nSamples)() # Os1
rgdSamples1=(c_double*nSamples)()  #Os2
hzAcq= c_double(10000)# Acquired data per sec

hdwf = c_int() #handle 
sts= c_byte() #AnalogIn status info
cAvailable = c_int()
cBufMax=c_int()
cLost = c_int()
cCorrupted = c_int()
fLost =0 
fCorrupted = 0

if OscilNum==3:
    print('초기화')
                
    dwf.FDwfAnalogOutReset(hdwf, WaveChannel)
    dwf.FDwfDeviceCloseAll()
#초기화하기 위함

#version = create_string_buffer(16)
#dwf.FDwfGetVersion(version)
#print("DWF Version: "+str(version.value))

#open device


k=0
repeat= int(input('반복할 횟수=?'))

MEANDATA=[]
MEANDATA1 = []

DEVDATA=[]
DEVDATA1=[]

VOLTDATA=[]
VOLTDATA1=[]



while k < repeat:
    print("Opening first device...")
    print('\t\t\t\t\t\t\t',k+1)

    dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))
    
    
    if hdwf.value == hdwfNone.value:
        print("failed to open device")
        break;
    if k ==0:
        dwf.FDwfAnalogOutNodeEnableSet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_bool(True))
    # Waveform
        f=input('waveform 을 입력하세요: funcSine,funcDC funcTriangle...\n')    
        offset=float(input('Offset 을 입력하세요: V\n'))

        if f=='funcSine':        
            dwf.FDwfAnalogOutNodeFunctionSet(hdwf, WaveChannel, AnalogOutNodeCarrier, funcSine)
            # Hz
            Hz=float(input('Frequency 를 입력하세요: Hz\n'))
            dwf.FDwfAnalogOutNodeFrequencySet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Hz))
            #Amplitude
            Amp=float(input('Amplitude 를 입력하세요: V\n'))
            dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Amp))
            print(f, Hz,'Hz', Amp,'V Amplitude', offset,'V offset')
        elif f=='funcTriangle':
            dwf.FDwfAnalogOutNodeFunctionSet(hdwf, WaveChannel, AnalogOutNodeCarrier, funcTriangle)
            # Hz
            Hz=float(input('Frequency 를 입력하세요: Hz\n'))
            dwf.FDwfAnalogOutNodeFrequencySet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Hz))
            #Amplitude
            Amp=float(input('Amplitude 를 입력하세요: V\n'))
            dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Amp))
            print(f, Hz,'Hz', Amp,'V Amplitude', offset,'V offset')
        elif f == 'funcSquare':
            dwf.FDwfAnalogOutNodeFunctionSet(hdwf, WaveChannel, AnalogOutNodeCarrier, funcSquare)
            # Hz
            Hz=float(input('Frequency 를 입력하세요: Hz\n'))
            dwf.FDwfAnalogOutNodeFrequencySet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Hz))
            #Amplitude
            Amp=float(input('Amplitude 를 입력하세요: V\n'))
            dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Amp))
            print(f, Hz,'Hz', Amp,'V Amplitude', offset,'V offset')
        elif f == 'funcDC':  
            dwf.FDwfAnalogOutNodeFunctionSet(hdwf, WaveChannel, AnalogOutNodeCarrier, funcDC)
            print(f, offset,'V offset')
        else:
            print("can't do that yet.. sorry")
            break;
        
    dwf.FDwfAnalogOutNodeOffsetSet(hdwf,WaveChannel,AnalogOutNodeCarrier, c_double(offset))

    if (OscilNum == 0) or (OscilNum == 2):
        dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(0), c_bool(True))
        dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(-1), c_double(10))
        dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
        dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
        dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(nSamples/hzAcq.value)) # -1 infinite record length
    if (OscilNum == 1) or (OscilNum == 2):
        dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(1), c_bool(True))
        dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(10))
        dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
        dwf.FDwfAnalogInFrequencySet(hdwf, hzAcq)
        dwf.FDwfAnalogInRecordLengthSet(hdwf, c_double(nSamples/hzAcq.value)) # -1 infinite record length
#    dwf.FDwfAnalogInChannelOffsetSet(hdwf, c_int(0), c_double(0))
#    dwf.FDwfAnalogInChannelOffsetSet(hdwf, c_int(1), c_double(0))
    print("Generating")
    # time = int(input('recording time? :s'))
    dwf.FDwfAnalogOutConfigure(hdwf,WaveChannel,c_bool(True))
    if k==0:
        time.sleep(2)
    else:
        time.sleep(0.01)
    print("Starting oscilloscope")
    dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_bool(True))
#    time.sleep(1)
    cSamples = 0
    

    a=time.time()
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
        if OscilNum == 0:
            dwf.FDwfAnalogInStatusData(hdwf, c_int(OscilNum), byref(rgdSamples, sizeof(c_double)*cSamples), cAvailable) # get channel 1 data
        elif OscilNum == 1:
            dwf.FDwfAnalogInStatusData(hdwf, c_int(OscilNum), byref(rgdSamples1, sizeof(c_double)*cSamples), cAvailable) # get channel 1 data
        else:
            dwf.FDwfAnalogInStatusData(hdwf, c_int(0), byref(rgdSamples, sizeof(c_double)*cSamples), cAvailable) # get channel 1 data
            dwf.FDwfAnalogInStatusData(hdwf, c_int(1), byref(rgdSamples1, sizeof(c_double)*cSamples), cAvailable) # get channel 1 data

        cSamples += cAvailable.value
    dwf.FDwfAnalogOutReset(hdwf, WaveChannel)
    dwf.FDwfDeviceCloseAll()    
    
    b=time.time()
    print('record {} secs'.format(float(b-a)))
    

    
    print("Recording done")
    if fLost:
        print("Samples were lost! Reduce frequency")
    if fCorrupted:
        print("Samples could be corrupted! Reduce frequency")

    summation=0
    summation1=0
    
    var=0
    var1=0
    
    Mean=0
    Mean1=0
    
    Var=0
    Var1=0
    
    dev=0
    dev1=0
    
    
    if (OscilNum == 0) or (OscilNum == 2):
        f = open("record.csv", "w")
        for v in rgdSamples:
            f.write("%s\n" % v)
        f.close()
        
        with open('record.csv') as data:
            Data = data.readlines()
        
        for i in range(len(rgdSamples)):
            summation+=float(Data[i])
        
        plt.plot(numpy.fromiter(rgdSamples, dtype = numpy.float))
        plt.show()

    if (OscilNum == 1) or (OscilNum == 2):
        f = open("record1.csv", "w")
        for v in rgdSamples1:
            f.write("%s\n" % v)
        f.close()
        with open('record1.csv') as data1:
            Data1 = data1.readlines()
        
        for i in range(len(rgdSamples1)):
            summation1+=float(Data1[i])
        plt.plot(numpy.fromiter(rgdSamples1, dtype = numpy.float))
        plt.show()
    
    
    
    if (OscilNum == 0) or (OscilNum == 2):
        Mean = summation/len(rgdSamples)
        
        for j in range(len(rgdSamples)):
            var+=(float(Data[j])-Mean)**2
        Var = var/len(rgdSamples)
        dev= numpy.sqrt(Var)
        VOLTDATA.append(offset)
        MEANDATA.append(Mean)
        DEVDATA.append(dev)
    
    if (OscilNum == 1) or (OscilNum == 2):
        Mean1 = summation1/len(rgdSamples1)
        
        for j in range(len(rgdSamples1)):
            var1+=(float(Data1[j])-Mean1)**2
        Var1 = var1/len(rgdSamples1)
        dev1= numpy.sqrt(Var1)
        VOLTDATA1.append(offset)
        MEANDATA1.append(Mean1)
        DEVDATA1.append(dev1)
    print('Mean',Mean)
    print('Mean1',Mean1)
    #print('Deviation',dev)
    print('V',offset)
    
    #phase
    mmax = np.max(rgdSamples)
    mmax1 = np.max(rgdSamples1)
    ind, ind1 = 0, 0
    i, j = 0, 0
    while(1):
        if(rgdSamples[i] == mmax):
            ind = i
            break;
        i = i + 1
    while(1):
        if(rgdSamples1[j] == mmax1):
            ind1 = j
            break;
        j = j + 1
    
    diff = np.abs(ind - ind1)
    delt = diff / 10000
    print(delt * Hz * 2 * np.pi)
    
    k+=1
    
# os.chdir('C:/Users/saurs/OneDrive/Desktop/EM/20190917/DATA')#DATA 폴더 따로 지정해서 저장

timestr= time.strftime("%m%d-%H%M")
if (OscilNum == 0) or (OscilNum == 2):
    #write_txt(VOLTDATA, 'voltdata'+timestr+'.txt',sep=' ')
    #write_txt(MEANDATA, 'Os1meandata'+timestr+'.txt',sep=' ')
    #write_txt(DEVDATA, 'Os1devdata'+timestr+'.txt',sep=' ')
    write_txt(Data, 'Alldata'+timestr+'.txt',sep=' ')
if (OscilNum == 1) or (OscilNum == 2):
    #write_txt(VOLTDATA1, 'voltdata1'+timestr+'.txt',sep=' ')
    #write_txt(MEANDATA1, 'Os2meandata'+timestr+'.txt',sep=' ')
    #write_txt(DEVDATA1, 'Os2devdata'+timestr+'.txt',sep=' ')
    write_txt(Data1, 'Alldata1'+timestr+'.txt',sep=' ')
    
    
#plt.plot(numpy.fromiter(rgdSamples, dtype = numpy.float))
#plt.show()
#%% Thermistor
import numpy as np
o11=np.loadtxt('Os2meandata1004-1428.txt') # 같이
o12=np.loadtxt('Os2meandata1004-1435.txt')
o13=np.loadtxt('Os2meandata1004-1438.txt')
#Volt= np.loadtxt('voltdata11002-0040.txt')  
o14=np.loadtxt('Os2meandata1004-1442.txt')
o15=np.loadtxt('Os2meandata1004-1445.txt')
o16=np.loadtxt('Os2meandata1004-1508.txt')
o17=np.loadtxt('Os2meandata1004-1512.txt')
o18=np.loadtxt('Os2meandata1004-1614.txt')
o19=np.loadtxt('Os2meandata1004-1601.txt')
o20=np.loadtxt('Os2meandata1004-1608.txt')
o21=np.loadtxt('Os2meandata1004-1611.txt')
o22=np.loadtxt('Os2meandata1004-1617.txt')
o23=np.loadtxt('Os2meandata1004-1622.txt')


x1=np.loadtxt('Os2meandata1001-2256.txt') # ㄸㄹ
x2=np.loadtxt('Os2meandata1001-2258.txt')
x3=np.loadtxt('Os2meandata1001-2259.txt')


ME=(x1[10]+x2[10]+x3[10])/3

z1=-np.mean(o11)+ME
z2=-np.mean(o12)+ME
z3=-np.mean(o13)+ME
z4=-np.mean(o14)+ME
z5=-np.mean(o15)+ME
z6=-np.mean(o16)+ME
z7=-np.mean(o17)+ME
z8=-np.mean(o18)+ME
z9=-np.mean(o19)+ME
z10=-np.mean(o20)+ME
z11=-np.mean(o21)+ME
z12=-np.mean(o22)+ME
z13=-np.mean(o23)+ME

z=[z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13]
o=[o11,o12,o13,o14,o15,o16,o17,o18,o19,o20,o21,o22,o23]
T=[25,35,100,110,120,10,20,30,40,50,60,90,180]
R=np.zeros(13)
for i in range(13):
    R[i]=z[i]/np.mean(o[i]) * 1E6
print(R)
plt.plot(T,R,'o')

#%% Histogram
import numpy as np
x1=np.loadtxt('Os1meandata1004-1853.txt') # 200 1번 Os
x2=np.loadtxt('Os1meandata1004-1847.txt') # 2000
x3=np.loadtxt('Os1meandata1004-1907.txt') # 20000
x4=np.loadtxt('AllData1004-1649.txt')

y1=np.loadtxt('Os2meandata1004-1853.txt')# 200 2번 os
y2=np.loadtxt('Os2meandata1004-1847.txt')# 2000
y3=np.loadtxt('Os2meandata1004-1907.txt')# 20000
y4=np.loadtxt('AllData11004-1649.txt')# 200000
num_bins = 15
n,bins,patches = plt.hist(x1[1:],num_bins)
#plt.xlabel
plt.show()
n,bins,patches = plt.hist(x2[1:],num_bins)
plt.show()
n,bins,patches = plt.hist(x3[1:],num_bins)
plt.show()
n,bins,patches = plt.hist(x4[1:],num_bins)
plt.show()

n,bins,patches = plt.hist(y1[1:],num_bins)
plt.show()
n,bins,patches = plt.hist(y2[1:],num_bins)
plt.show()
n,bins,patches = plt.hist(y3[1:],num_bins)
plt.show()
n,bins,patches = plt.hist(y4[1:],num_bins)
plt.show()
#%%
z=x11-x1
#z1=x11-x1
#z2=x12-x2
#z3=x13-x3
print(z)
Resist=z/x11 * 8200
#Resist= z[40:200]/x11[40:200] * 1E6
#resist=np.delete(Resist,6)
#X=np.delete(x,6)
plt.plot(Volt,Resist,'ro')
plt.show()
plt.plot(z,x11/8200,'bo' )
plt.show()

#%%

x=z
from scipy.odr import *

def lin_func(p,x):
    return p[0] + p[1] * x 
lin_model=Model(lin_func)
data=RealData(z,x11/8200)
odr=ODR(data, lin_model,beta0=[0.,1.0])
out=odr.run()
#out.pprint()
out.beta
print(out.sd_beta)
print(out.sum_square)
plt.plot(x,lin_func(out.beta,x))
plt.plot(z,x11/8200,'bo')
plt.show()
print(out.beta[1],'x' , out.beta[0])
print('resistance ={}'.format(1/out.beta[1]))

#%%
x11=np.loadtxt('Os2meandata1001-2330.txt')
Volt= np.loadtxt('voltdata11001-2330.txt')  

x1=np.loadtxt('Os2meandata1001-2351.txt') 
z=x1-x11
#plt.plot(x11,z,'ro')
r1= z/x1 *1E6
plt.plot(x11[30:200],r1[30:200],'bo')
print(np.mean(r1[190:200]))
#%%
I = x1/1E6 
plt.plot(I,z,'b')
#%%
a=np.loadtxt('Os2meandata1001-2301.txt')
a1=np.loadtxt('Os2meandata1001-2256.txt')

z= (a1-a)/a1
print(z)