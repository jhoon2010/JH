import os

def write_txt(list,fname,sep):

    file=open(fname,'w')
#    vstr=''
    
#    for a in list:
#        vstr = vstr + str(a) +sep 
#    vstr = vstr.rstrip(sep)
    
    file.writelines(list)
    file.close()
    print('[파일 저장 완료]')
#Ref: https://data-make.tistory.com/109

#%%
from ctypes import *
import numpy
import time
import sys
from datetime import datetime
import math
import matplotlib.pyplot as plt
import os 
os.chdir('C:/Users/ZEST/Desktop/1')#import 장소.
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

nSamples = int(1000000)
rgdSamples=(c_double*nSamples)() # Os1
rgdSamples1=(c_double*nSamples)()  #Os2
hzAcq= c_double(1000000)# Acquired data per sec

hdwf = c_int() #handle 
sts= c_byte() #AnalogIn status info
cAvailable = c_int()
cBufMax=c_int()
cLost = c_int()
cCorrupted = c_int()
fLost = 0 
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

func=input('waveform 을 입력하세요: funcSine,funcDC funcTriangle...\n')    
Hz = 10000
Amp = 3
offset = 0
while Hz < 10001:
    print("Opening first device...")
    print(Hz)
    print('\t\t\t\t\t\t\t',k+1)

    dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))
    
    
    if hdwf.value == hdwfNone.value:
        print("failed to open device")
        break;
    if k>=0:
        dwf.FDwfAnalogOutNodeEnableSet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_bool(True))
    # Waveform
        # f=input('waveform 을 입력하세요: funcSine,funcDC funcTriangle...\n')    
        # offset=float(input('Offset 을 입력하세요: V\n'))

        if func=='funcSine':        
            dwf.FDwfAnalogOutNodeFunctionSet(hdwf, WaveChannel, AnalogOutNodeCarrier, funcSine)
            # Hz
            # Hz=float(input('Frequency 를 입력하세요: Hz\n'))
            dwf.FDwfAnalogOutNodeFrequencySet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Hz))
            #Amplitude
            # Amp=float(input('Amplitude 를 입력하세요: V\n'))
            dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Amp))
            print(func, Hz,'Hz', Amp,'V Amplitude', offset,'V offset')
        elif func=='funcTri':
            dwf.FDwfAnalogOutNodeFunctionSet(hdwf, WaveChannel, AnalogOutNodeCarrier, funcTriangle)
            # Hz
            # Hz=float(input('Frequency 를 입력하세요: Hz\n'))
            dwf.FDwfAnalogOutNodeFrequencySet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Hz))
            #Amplitude
            # Amp=float(input('Amplitude 를 입력하세요: V\n'))
            dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Amp))
            print(func, Hz,'Hz', Amp,'V Amplitude', offset,'V offset')
        elif func == 'funcSqu':
            dwf.FDwfAnalogOutNodeFunctionSet(hdwf, WaveChannel, AnalogOutNodeCarrier, funcSquare)
            # Hz
            # Hz=float(input('Frequency 를 입력하세요: Hz\n'))
            dwf.FDwfAnalogOutNodeFrequencySet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Hz))
            #Amplitude
            # Amp=float(input('Amplitude 를 입력하세요: V\n'))
            dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, WaveChannel, AnalogOutNodeCarrier, c_double(Amp))
            print(func, Hz,'Hz', Amp,'V Amplitude', offset,'V offset')
        elif func == 'funcDC':  
            dwf.FDwfAnalogOutNodeFunctionSet(hdwf, WaveChannel, AnalogOutNodeCarrier, funcDC)
            print(func, offset,'V offset')
        else:
            print("can't do that yet.. sorry")
            dwf.FDwfAnalogOutReset(hdwf, WaveChannel)
            dwf.FDwfDeviceCloseAll()
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
    if k>=0:
        time.sleep(1)
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
#           
#        for i in range(len(rgdSamples)):
#           summation+=float(Data[i])
# Oscil 1 -> blue
        plt.plot(numpy.fromiter(rgdSamples, dtype = numpy.float),'b')

    if (OscilNum == 1) or (OscilNum == 2):
        f = open("record1.csv", "w")
        for v in rgdSamples1:
            f.write("%s\n" % v)
        f.close()
        with open('record1.csv') as data1:
            Data1 = data1.readlines()
        
#        for i in range(len(rgdSamples1)):
#            summation1+=float(Data1[i])
# Oscil 2 -> red            
        plt.plot(numpy.fromiter(rgdSamples1, dtype = numpy.float),'r')
        plt.show()
    
    
    
#    if (OscilNum == 0) or (OscilNum == 2):
#        Mean = summation/len(rgdSamples)
        
#        for j in range(len(rgdSamples)):
#            var+=(float(Data[j])-Mean)**2
#        Var = var/len(rgdSamples)
#        dev= numpy.sqrt(Var)
#        VOLTDATA.append(offset)
#        MEANDATA.append(Mean)
#        DEVDATA.append(dev)
    
#    if (OscilNum == 1) or (OscilNum == 2):
#        Mean1 = summation1/len(rgdSamples1)
        
#        for j in range(len(rgdSamples1)):
#            var1+=(float(Data1[j])-Mean1)**2
#        Var1 = var1/len(rgdSamples1)
#        dev1= numpy.sqrt(Var1)
#        VOLTDATA1.append(offset)
#        MEANDATA1.append(Mean1)
#        DEVDATA1.append(dev1)
#    print('Mean',Mean)
#    print('Mean1',Mean1)
#    print('Deviation',dev)
#    print('V',offset)
    
# phase
#    mmax = numpy.max(rgdSamples)
#    mmax1 = numpy.max(rgdSamples1)
#    ind, ind1 = 0, 0
#    i, j = 0, 0
#    while(1):
#        if(rgdSamples[i] == mmax):
#            ind = i
#            break;
#        i = i + 1
#    while(1):
#        if(rgdSamples1[j] == mmax1):
#            ind1 = j
#            break;
#        j = j + 1
    
#    diff = numpy.abs(ind - ind1)
#    delt = diff / 10000
    
#    print(delt * 2 * numpy.pi)
    #timestr= time.strftime("%m%d-%H%M")
    if (OscilNum == 0) or (OscilNum == 2):
    #write_txt(VOLTDATA, 'voltdata'+timestr+'.txt',sep=' ')
    #write_txt(MEANDATA, 'Os1meandata'+timestr+'.txt',sep=' ')
    #write_txt(DEVDATA, 'Os1devdata'+timestr+'.txt',sep=' ')
    #write_txt(Data, 'Alldata'+timestr+'.txt',sep=' ')
        write_txt(Data, 'Ex2R_1'+func+str(Hz)+'.txt',sep=' ')

    if (OscilNum == 1) or (OscilNum == 2):
        write_txt(Data1, 'Ex2R_2'+func+str(Hz)+'.txt',sep=' ')
    #write_txt(VOLTDATA1, 'voltdata1'+timestr+'.txt',sep=' ')
    #write_txt(MEANDATA1, 'Os2meandata'+timestr+'.txt',sep=' ')
    #write_txt(DEVDATA1, 'Os2devdata'+timestr+'.txt',sep=' ')
    #write_txt(Data1, 'Alldata1'+timestr+'.txt',sep=' ')
    
    Hz = Hz + 50
    k+=1
    
# os.chdir('C:/Users/saurs/OneDrive/Desktop/EM/20190917/DATA')#DATA 폴더 따로 지정해서 저장

#timestr= time.strftime("%m%d-%H%M")
#if (OscilNum == 0) or (OscilNum == 2):
    #write_txt(VOLTDATA, 'voltdata'+timestr+'.txt',sep=' ')
    #write_txt(MEANDATA, 'Os1meandata'+timestr+'.txt',sep=' ')
    #write_txt(DEVDATA, 'Os1devdata'+timestr+'.txt',sep=' ')
    #write_txt(Data, 'Alldata'+timestr+'.txt',sep=' ')
#    write_txt(Data, 'Ex2'+'_'+str(f)+'_'+str(Hz)+'.txt',sep=' ')

#if (OscilNum == 1) or (OscilNum == 2):
#    write_txt(Data1, 'Ex2'+'_'+str(f)+'_'+str(Hz)+'.txt',sep=' ')
    #write_txt(VOLTDATA1, 'voltdata1'+timestr+'.txt',sep=' ')
    #write_txt(MEANDATA1, 'Os2meandata'+timestr+'.txt',sep=' ')
    #write_txt(DEVDATA1, 'Os2devdata'+timestr+'.txt',sep=' ')
    #write_txt(Data1, 'Alldata1'+timestr+'.txt',sep=' ')
    
    
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
import os
import numpy as np
import matplotlib.pyplot as plt
os.chdir('C:/Users/ZEST/Desktop/1')#import 장소.

leng = [1000000,333330,111110,37000,13000,10000,6700,5000,4200,4000,3300,3000,2500,2300,2000,1800,1600,1550,1400,1370,1330,1250,1200,1100,1100,500,180,100]


a1=np.loadtxt('Ex2R_1funcsine1.txt',skiprows=0,max_rows=leng[0])
a2=np.loadtxt('Ex2R_1funcsine3.txt',skiprows=leng[1],max_rows=leng[1])
a3=np.loadtxt('Ex2R_1funcsine9.txt',skiprows=leng[2],max_rows=leng[2])
a4=np.loadtxt('Ex2R_1funcsine27.txt',skiprows=leng[3],max_rows=leng[3])
a5=np.loadtxt('Ex2R_1funcsine81.txt',skiprows=leng[4],max_rows=leng[4])
a6=np.loadtxt('Ex2R_1funcsine100.txt',skiprows=leng[5],max_rows=leng[5])
a7=np.loadtxt('Ex2R_1funcsine150.txt',skiprows=leng[6],max_rows=leng[6])
a8=np.loadtxt('Ex2R_1funcsine200.txt',skiprows=leng[7],max_rows=leng[7])
a9=np.loadtxt('Ex2R_1funcsine243.txt',skiprows=leng[8],max_rows=leng[8])
a10=np.loadtxt('Ex2R_1funcsine250.txt',skiprows=leng[9],max_rows=leng[9])
a11=np.loadtxt('Ex2R_1funcsine300.txt',skiprows=leng[10],max_rows=leng[10])
a12=np.loadtxt('Ex2R_1funcsine350.txt',skiprows=leng[11],max_rows=leng[11])
a13=np.loadtxt('Ex2R_1funcsine400.txt',skiprows=leng[12],max_rows=leng[12])
a14=np.loadtxt('Ex2R_1funcsine450.txt',skiprows=leng[13],max_rows=leng[13])
a15=np.loadtxt('Ex2R_1funcsine500.txt',skiprows=leng[14],max_rows=leng[14])
a16=np.loadtxt('Ex2R_1funcsine550.txt',skiprows=leng[15],max_rows=leng[15])
a17=np.loadtxt('Ex2R_1funcsine600.txt',skiprows=leng[16],max_rows=leng[16])
a18=np.loadtxt('Ex2R_1funcsine650.txt',skiprows=leng[17],max_rows=leng[17])
a19=np.loadtxt('Ex2R_1funcsine700.txt',skiprows=leng[18],max_rows=leng[18])
a20=np.loadtxt('Ex2R_1funcsine729.txt',skiprows=leng[19],max_rows=leng[19])
a21=np.loadtxt('Ex2R_1funcsine750.txt',skiprows=leng[20],max_rows=leng[20])
a22=np.loadtxt('Ex2R_1funcsine800.txt',skiprows=leng[21],max_rows=leng[21])
a23=np.loadtxt('Ex2R_1funcsine850.txt',skiprows=leng[22],max_rows=leng[22])
a24=np.loadtxt('Ex2R_1funcsine900.txt',skiprows=leng[23],max_rows=leng[23])
a25=np.loadtxt('Ex2R_1funcsine950.txt',skiprows=leng[24],max_rows=leng[24])
a26=np.loadtxt('Ex2R_1funcsine2187.txt',skiprows=leng[25],max_rows=leng[25])
a27=np.loadtxt('Ex2R_1funcsine6561.txt',skiprows=leng[26],max_rows=leng[26])



b1=np.loadtxt('Ex2R_2funcsine1.txt',skiprows=0,max_rows=leng[0])
b2=np.loadtxt('Ex2R_2funcsine3.txt',skiprows=leng[1],max_rows=leng[1])
b3=np.loadtxt('Ex2R_2funcsine9.txt',skiprows=leng[2],max_rows=leng[2])
b4=np.loadtxt('Ex2R_2funcsine27.txt',skiprows=leng[3],max_rows=leng[3])
b5=np.loadtxt('Ex2R_2funcsine81.txt',skiprows=leng[4],max_rows=leng[4])
b6=np.loadtxt('Ex2R_2funcsine100.txt',skiprows=leng[5],max_rows=leng[5])
b7=np.loadtxt('Ex2R_2funcsine150.txt',skiprows=leng[6],max_rows=leng[6])
b8=np.loadtxt('Ex2R_2funcsine200.txt',skiprows=leng[7],max_rows=leng[7])
b9=np.loadtxt('Ex2R_2funcsine243.txt',skiprows=leng[8],max_rows=leng[8])
b10=np.loadtxt('Ex2R_2funcsine250.txt',skiprows=leng[9],max_rows=leng[9])
b11=np.loadtxt('Ex2R_2funcsine300.txt',skiprows=leng[10],max_rows=leng[10])
b12=np.loadtxt('Ex2R_2funcsine350.txt',skiprows=leng[11],max_rows=leng[11])
b13=np.loadtxt('Ex2R_2funcsine400.txt',skiprows=leng[12],max_rows=leng[12])
b14=np.loadtxt('Ex2R_2funcsine450.txt',skiprows=leng[13],max_rows=leng[13])
b15=np.loadtxt('Ex2R_2funcsine500.txt',skiprows=leng[14],max_rows=leng[14])
b16=np.loadtxt('Ex2R_2funcsine550.txt',skiprows=leng[15],max_rows=leng[15])
b17=np.loadtxt('Ex2R_2funcsine600.txt',skiprows=leng[16],max_rows=leng[16])
b18=np.loadtxt('Ex2R_2funcsine650.txt',skiprows=leng[17],max_rows=leng[17])
b19=np.loadtxt('Ex2R_2funcsine700.txt',skiprows=leng[18],max_rows=leng[18])
b20=np.loadtxt('Ex2R_2funcsine729.txt',skiprows=leng[19],max_rows=leng[19])
b21=np.loadtxt('Ex2R_2funcsine750.txt',skiprows=leng[20],max_rows=leng[20])
b22=np.loadtxt('Ex2R_2funcsine800.txt',skiprows=leng[21],max_rows=leng[21])
b23=np.loadtxt('Ex2R_2funcsine850.txt',skiprows=leng[22],max_rows=leng[22])
b24=np.loadtxt('Ex2R_2funcsine900.txt',skiprows=leng[23],max_rows=leng[23])
b25=np.loadtxt('Ex2R_2funcsine950.txt',skiprows=leng[24],max_rows=leng[24])
b26=np.loadtxt('Ex2R_2funcsine2187.txt',skiprows=leng[25],max_rows=leng[25])
b27=np.loadtxt('Ex2R_2funcsine6561.txt',skiprows=leng[26],max_rows=leng[26])
b28=np.loadtxt('Ex2R_2funcsine10000.txt',skiprows=leng[27],max_rows=leng[27])


m1=np.argmax(a1)
m2=np.argmax(a2)
m3=np.argmax(a3)
m4=np.argmax(a4)
m5=np.argmax(a5)
m6=np.argmax(a6)
m7=np.argmax(a7)
m8=np.argmax(a8)
m9=np.argmax(a9)
m10=np.argmax(a10)
m11=np.argmax(a11)
m12=np.argmax(a12)
m13=np.argmax(a13)
m14=np.argmax(a14)
m15=np.argmax(a15)
m16=np.argmax(a16)
m17=np.argmax(a17)
m = [m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17]
n1=np.max(b1)
n2=np.max(b2)
n3=np.max(b3)
n4=np.max(b4)
n5=np.max(b5)
n6=np.max(b6)
n7=np.max(b7)
n8=np.max(b8)
n9=np.max(b9)
n10=np.max(b10)
n11=np.max(b11)
n12=np.max(b12)
n13=np.max(b13)
n14=np.max(b14)
n15=np.max(b15)
n16=np.max(b16)
n17=np.max(b17)
n18=np.max(b18)
n19=np.max(b19)
n20=np.max(b20)
n21=np.max(b21)
n22=np.max(b22)
n23=np.max(b23)
n24=np.max(b24)
n25=np.max(b25)
n26=np.max(b26)
n27=np.max(b27)
n28=np.max(b28)
n = [n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25,n26,n27,n28]

plt.plot(n)

#d = np.zeros(17)
#for i in range(0, 17):
#    d[i] = m[i] - n[i]

#print(d)
#delta = np.zeros(17)
#for i in range(0, 17):
#    delta[i] = d[i] / leng[i] * 2 *np.pi * 57.3
#print(delta)
#plt.plot(delta)