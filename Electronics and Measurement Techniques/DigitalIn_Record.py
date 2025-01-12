"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2018-07-23

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
from dwfconstants import *
import math
import sys
import matplotlib.pyplot as plt
import numpy


if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

hdwf = c_int()
sts = c_ubyte()

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

print("Opening first device")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == 0:
    print("failed to open device")
    szerr = create_string_buffer(512)
    dwf.FDwfGetLastErrorMsg(szerr)
    print(str(szerr.value))
    quit()

print("Configuring Digital Out / In...")

# generate counter
for i in range(0, 16):
    dwf.FDwfDigitalOutEnableSet(hdwf, c_int(i), c_int(1))
    dwf.FDwfDigitalOutDividerSet(hdwf, c_int(i), c_int(1<<i))
    dwf.FDwfDigitalOutCounterSet(hdwf, c_int(i), c_int(1000), c_int(1000))

dwf.FDwfDigitalOutConfigure(hdwf, c_int(1))

# set number of sample to acquire
nSamples = 100000
rgwSamples = (c_uint16*nSamples)()
cAvailable = c_int()
cLost = c_int()
cCorrupted = c_int()
iSample = 0
fLost = 0
fCorrupted = 0
hzDI = c_double()

dwf.FDwfDigitalInInternalClockInfo(hdwf, byref(hzDI))
print("DigitanIn base freq: "+str(hzDI.value))

# in record mode samples after trigger are acquired only
dwf.FDwfDigitalInAcquisitionModeSet(hdwf, acqmodeRecord)
# sample rate = system frequency / divider, 100MHz/1000 = 100kHz
dwf.FDwfDigitalInDividerSet(hdwf, c_int(int(hzDI.value/100e3)))
# 16bit per sample format
dwf.FDwfDigitalInSampleFormatSet(hdwf, c_int(16))
# number of samples after trigger
dwf.FDwfDigitalInTriggerPositionSet(hdwf, c_int(int(nSamples/2))) # trigger in the middle
# number of samples before trigger
dwf.FDwfDigitalInTriggerPrefillSet(hdwf, c_int(nSamples-int(nSamples/2)))
# trigger when all digital pins are low
dwf.FDwfDigitalInTriggerSourceSet(hdwf, trigsrcDetectorDigitalIn)
# trigger detector mask:                  low &   hight & ( rising | falling )
dwf.FDwfDigitalInTriggerSet(hdwf, c_int(0xFFFF), c_int(0), c_int(0), c_int(0))
# for Digital Discovery bit order: DIO24:39; with 32 bit sampling [DIO24:39 + DIN0:15]
dwf.FDwfDigitalInInputOrderSet(hdwf, c_int(1))
# begin acquisition
dwf.FDwfDigitalInConfigure(hdwf, c_bool(0), c_bool(1))

print("Recording...")


while True:
    dwf.FDwfDigitalInStatus(hdwf, c_int(1), byref(sts))
    dwf.FDwfDigitalInStatusRecord(hdwf, byref(cAvailable), byref(cLost), byref(cCorrupted))
    
    iSample += cLost.value
    iSample %= nSamples
    
    if cLost.value :
        fLost = 1
    if cCorrupted.value :
        fCorrupted = 1

    iBuffer = 0
    while cAvailable.value>0:
        cSamples = cAvailable.value
        # we are using circular sample buffer, prevent overflow
        if iSample+cAvailable.value > nSamples:
            cSamples = nSamples-iSample
        dwf.FDwfDigitalInStatusData2(hdwf, byref(rgwSamples, 2*iSample), c_int(iBuffer), c_int(2*cSamples))
        iBuffer += cSamples
        cAvailable.value -= cSamples
        iSample += cSamples
        iSample %= nSamples

    if sts.value == DwfStateDone.value :
        break

dwf.FDwfDeviceClose(hdwf)

if iSample != 0 :
    rgwSamples = rgwSamples[iSample:]+rgwSamples[:iSample]

print("  done")
if fLost:
    print("Samples were lost! Reduce sample rate")
if cCorrupted:
    print("Samples could be corrupted! Reduce sample rate")

f = open("record.csv", "w")
for v in rgwSamples:
    f.write("%s\n" % v)
f.close()

plt.plot(numpy.fromiter(rgwSamples, dtype = numpy.uint16))
plt.show()
