#import serial
#import serial.tools.list_ports
import matplotlib.pyplot as plt
import numpy as np
#from drawnow import *

# Settings
FS = 250
T = 10
FILTER_F1 = 45
FILTER_F2 = 70
'''
# SEARCH FOR THE PORT
#port = serial.tools.list_ports.comports().pop()
portName = 'COM5'#port.device

# OPEN AND READ
ser = serial.Serial(portName, 9600)

def record():
	global ser
	data = []
	for n in range(0, T):

		str = ser.readline()
		if str == b'!\r\n':
			value = 0
		else:
			value = int( str )
		data.append(value)
		print(n)
	return data
'''
def f2n(f):
	return int(T*f)


def filtered(data):
	data_f = np.fft.fft(data)
	data_f[0] = 0
	n1 = f2n(FILTER_F1)
	n2 = f2n(FILTER_F2)
	data_f[n1:n2] = 0
	data_f[-n2:-n1] = 0
	data_fi= np.fft.ifft(data_f)
	return data_fi
'''
run = True
while run:
	data = record()
	data_filtered = filtered(data)



	plt.plot(data)
	plt.plot(data_filtered)
	plt.show()
	
	run = False

data_f = np.fft.fft(data_filtered)
data_fa = np.abs(data_f)
plt.plot(data_fa)
plt.show()
'''
