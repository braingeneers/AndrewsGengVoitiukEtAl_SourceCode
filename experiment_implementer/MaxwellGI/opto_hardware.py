#!pip install pyvisa
#!pip install ThorlabsPM100
#!pip install pyvisa-py
# https://machinelearningmastery.com/curve-fitting-with-python/


#Optogenetics: top level library
#Opto_hardware: low-level library
#optogentics_helper: old code

import numpy as np
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import csv
import pandas as pd

import pyvisa
from ThorlabsPM100 import ThorlabsPM100

import time
from datetime import datetime, timezone

import pytz
from pytz import timezone, utc

from pprint import pformat

import struct
from pySerialTransfer import pySerialTransfer as txfer


import math
np.random.seed(42)



class OptoHardware:
    """
    Opto Hardware Environment
    Environment for storing and calculating Opto Hardware parameters.
    """

    def __init__(self, verbose = True):
        """
        Constructs attributes for OptoHardware object.
        """
        self.verbose = verbose
        self.config_name = "None"
        self.LED_wavelength = None #nm
        self.LED_wavelength_color = '#00A9FF' #'#00C0FF'
        self.fiber_core_diam_um = None #um core
        self.fiber_numerical_aperture = None #NA
        self.fiber_len_mm = None #mm
        self.led_docs_max_output_mW_200core = None
        self.link_to_led_docs = None
        self.path_to_Thorlabs = None #'USB0::4883::32888::P0026693::0::INSTR'
        self.power_meter = None
        self.path_to_csv = None #''
        self.DAC_range = 4095 #default; 4096 max (overflows 5V cap)
        self.x = []
        self.y = []
        self.a,self.b,self.c = [0,0,0] # power function fit: y -> x;  y = a*x + b*x^2 + c
        self.a_inv,self.b_inv,self.c_inv = [0,0,0] # power function inverse: x -> y

        self.rm = pyvisa.ResourceManager()

        # Arduino-specific:
        self.arduino_path = None #'dev/tty/ACM0'
        self.ramp = False #ramp up signal 0 to max (over ramp_time duration) if True, square pulse if False
        self.ramp_time = 0 #ms, 1000ms = 1sec; Arduino expects o to max 2^16 (65536 ms)
        self.baud = 115200
        self.link = None #open_link(arduino_path, baud)
        self.use_maxwell = True
        self.arduino_reply = None
        self.arduino_intensity = 0

    def __str__(self):
        info = 'OptoHardware parameters for %s:\n' % self.config_name
        info += pformat(vars(self)) #'LED_wavelength: %d \n' % (self.LED_wavelength)
        info += '\n y = a * x + b * x^2 + c'
        info += '\n y = %.5f * x + %.5f * x^2 + %.5f' % (self.a, self.b, self.c)
        return info


    def list_USB_devices(self):
        print("Available USB devices:")
        print(pformat(self.rm.list_resources()))

    def use_maxwell_on(self):
        self.use_maxwell = True
        self.set_arduino_intensity(self.arduino_intensity)

    def use_maxwell_off(self):
        self.use_maxwell = False
        self.set_arduino_intensity(self.arduino_intensity)



    def init_ThorlabsPM100(self, path_to_Thorlabs):

        self.path_to_Thorlabs = path_to_Thorlabs #'USB0::4883::32888::P0026693::0::INSTR'

        inst = self.rm.open_resource(self.path_to_Thorlabs)
        self.power_meter = ThorlabsPM100.ThorlabsPM100(inst=inst)
        inst.timeout = None

        print("Measurement type :", self.power_meter.getconfigure)
        print("Current value :", self.power_meter.read)

        print(inst.query("*IDN?"))
        #print(rm)
        #print(inst)

        self.power_meter.sense.average.count = 100 # write property
        print(self.power_meter.sense.average.count) # read property

        if self.verbose: print(self.power_meter.read) # Read-only property
        self.power_meter.system.beeper.immediate() # method

        time.sleep(2)


    def close_ThorlabsPM100(self):
        self.rm.close()
        self.rm.visalib._registry.clear()


    def measure_power_output(self):
        y = []

        for intensity in self.x:
            self.set_arduino_intensity(intensity)
            print("Intensity: %0.5f \t DAC_bitvalue: %d \t Expected Voltage: %0.5f" % (intensity, self.get_DAC_bitvalue(intensity), 5*intensity), end='')
            time.sleep(1.5)
            measurement = self.power_meter.read
            self.power_meter.system.beeper.immediate() # method
            print("\t Measured [Watts]:", measurement) # Read-only property
            y.append(measurement)

        self.set_arduino_intensity(0)
        self.y = [element * 1000 for element in y] #Watts to mW

        print("Collected measurements [mW]:")
        print(self.y)



    def save_parameter_csv(self):
        """
        Save power output calibrations to csv file
        """
        self.path_to_csv = (datetime.now(tz=pytz.timezone('US/Pacific')).strftime('%Y%m%dT%H%M%S-')) + self.config_name.replace(' ', '-').lower() + ".csv"

        d = {'config_name': self.config_name,
            'path_to_csv': self.path_to_csv,
            'DAC_range': self.DAC_range,
            'baud': 115200,
            'LED_wavelength': self.LED_wavelength,
            'LED_wavelength_color': self.LED_wavelength_color,
            'led_docs_max_output_mW_200core': self.led_docs_max_output_mW_200core,
            'link_to_led_docs': self.link_to_led_docs,
            'fiber_core_diam_um': self.fiber_core_diam_um,
            'fiber_numerical_aperture': self.fiber_numerical_aperture,
            'fiber_len_mm': self.fiber_len_mm,
            'verbose': self.verbose,
            'ramp': self.ramp, #ramp up signal 0 to max (over ramp_time duration) if True, square pulse if False
            'ramp_time': self.ramp_time, #ms, 1000ms = 1sec; Arduino expects o to max 2^16 (65536 ms)
            'a': self.a,
            'b' : self.b,
            'c' : self.c,
            'x': None,
            'y': None}

        df = pd.DataFrame(data=d,  index=[0])
        df['x'] = df['x'].astype('object')
        df['y'] = df['y'].astype('object')
        df.at[0, 'x'] = self.x
        df.at[0, 'y'] = self.y


        df.to_csv(self.path_to_csv, index=False)
        if self.verbose: print("Saved to:", self.path_to_csv)



    def set_path_to_parameter_csv(self, path):
        self.path_to_csv = path
        if self.verbose: print("Using opto hardware parameters from:", self.path_to_csv)


    def load_parameter_csv(self):
        """
        Loads power output calibrations from saved csv file
        """

        df = pd.read_csv(self.path_to_csv)
        df.to_string(index=False)
        self.x = list(map(float, df.x[0][1:-1].split(',')))
        self.y = list(map(float, df.y[0][1:-1].split(',')))

        self.config_name = df.config_name[0]
        self.LED_wavelength = df.LED_wavelength[0]
        self.LED_wavelength_color = df.LED_wavelength_color[0]
        self.fiber_core_diam_um = df.fiber_core_diam_um[0]
        self.fiber_numerical_aperture = df.fiber_numerical_aperture[0]
        self.fiber_len_mm = df.fiber_len_mm[0]
        self.DAC_range = df.DAC_range[0]
        self.ramp =  df.ramp[0]  #ramp up signal 0 to max (over ramp_time duration) if True, square pulse if False
        self.ramp_time =  df.ramp_time[0] #ms
        self.a,self.b,self.c = [df.a[0],df.b[0],df.c[0]] # power function fit: y -> x;  y = a*x + b*x^2 + c
        if self.verbose: print("Loaded hardware parameters!")


    def plot_calibration_curve(self, x_raw, y_raw, title, xlabel, ylabel):

        # define the true objective function
        def objective(x, a, b, c):
            return a * x + b * x**2 + c



        # choose the input and output variables
        x, y = np.array(x_raw), np.array(y_raw)

        # curve fit
        popt, _ = curve_fit(objective, x, y)


        #fig, axs = plt.subplots(figsize=(8, 6))
        # We change the fontsize of minor ticks label

        # summarize the parameter values
        a, b, c = popt
        print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
        # plot input vs output
        plt.scatter(x, y)
        plt.rcParams.update({'font.size': 12})
        plt.title(title)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

        # define a sequence of inputs between the smallest and largest known inputs
        x_line = np.arange(min(x), max(x), 0.01)

        # calculate the output for the range
        y_line = objective(x_line, a, b, c)

        # create a line plot for the mapping function
        plt.plot(x_line, y_line, '--', color='red')
        plt.show()

        return a, b, c

        #bare_LED = 112.30053 * x + -28.80158 * x**2 + -0.52006 #bare LED
        #bare_LED_inv = return ((112.30053-math.sqrt(-115.20632*y + 12551.49483)) / (57.60316)) #bare LED
        #cannula = 0.55040 * x + -0.13691 * x**2 + -0.00290 # Cannula LED
        #cannula_inv = ((0.5504 - math.sqrt(-0.54764*y + 0.301352004)) / 0.27382) # Cannula LED

    def plot_power_curve(self):
        title = "Power of " + str(self.LED_wavelength) + "nm LED, " + str(self.fiber_core_diam_um) + "nm core cable & cannula, cannula " + str(self.fiber_len_mm) + "mm"
        xlabel = "Arduino DAC Modulation (Fraction of Total Voltage)"
        ylabel = "Power (mW)"
        a, b, c, = self.plot_calibration_curve(self.x, self.y, title,  xlabel, ylabel)
        self.a = a
        self.b = b
        self.c = c

    def plot_power_density_curve(self):
        title = "Power Density of " + str(self.LED_wavelength) + "nm LED, " + str(self.fiber_core_diam_um) + "nm core cable & cannula, cannula " + str(self.fiber_len_mm) + "mm"
        xlabel = "Arduino DAC Modulation (Fraction of Total Voltage)"
        ylabel = "Power Density (mW/mm$^2$)"
        y_densities = [self.power_density(y_val) for y_val in self.y]
        a, b, c, = self.plot_calibration_curve(self.x, y_densities, title,  xlabel, ylabel)


    # ---------
    def get_DAC_bitvalue(self, intensity_fraction):
        """
        Given intensity_fraction (power intensity fraction from 0 to 1)
        Returns corresponding bitvalue for DAC (range 0 to self.DAC_range)
        """
        if not (0 <= intensity_fraction <= 1):
            raise ValueError("Error: Input must be in range 0 to 1")

        return int(round(self.DAC_range * intensity_fraction))

    def get_LED_theoretical_max_power(self):
        return self.power_density(self.led_max_output_mW_from_thorlabs_docs)


    def average_power(self, power_density):
        """
        Given power_density (mW/mm^2, based on optical fiber core diameter)
        Returns average_power (mW, raw power density recorded on Thorlabs meter)
        """
        beam_diameter = self.fiber_core_diam_um * 10**-3 #convert to mm
        average_power = power_density * (((beam_diameter/2)**2)* math.pi ) #[mW]
        return average_power


    def power_density(self, p_avg_mW):
        """
        Given p_avg_mW (mW, raw power recorded on Thorlabs meter)
        Returns power_density (mW/mm^2) based on optical fiber core diameter
        """
        average_power = p_avg_mW # 0.5 #122 *10**-3 #25 * 10**-3# mW
        beam_diameter = self.fiber_core_diam_um * 10**-3 #convert to mm
        power_density = ((average_power)) / (((beam_diameter/2)**2)* math.pi  )  #[mW/mm²]
        return power_density



    def arduino_setting_to_power(self, x): #get_power_from_arduino_setting
        """
        Given x (power intensity fraction from 0 to 1)
        Returns y (power in [mW])
        """
        if not (min(self.x) <= x <= max(self.x)):
            raise ValueError("Error: Input must be in range {:.4f} to {:.4f}".format(min(self.x), max(self.x)))
        y = self.a * x + self.b * x**2 + self.c
        return 0 if y < 0 else y


    def power_to_arduino_setting(self, y): #get_arduino_setting_from_power
        """
        Given y (power in [mW])
        Returns x (power intensity fraction from 0 to 1)
        """
        if not (0 <= y <= max(self.y)):
            raise ValueError("Error: Input must be in range ", min(self.y), "to", max(self.y))

        x_value = np.interp(y, self.y, self.x)
        #x_value = ((0.5504 - math.sqrt(-0.54764*y + 0.301352004)) / 0.27382)
        return x_value


    def arduino_setting_to_power_density(self, percent_intensity):
        """
         Calculates power density [mW/mm^2] from arduino setting [fraction from 0 to 1]
        """
        mW_density = self.power_density(self.arduino_setting_to_power(percent_intensity))
        return mW_density


    def power_density_to_arduino_setting(self, mW_density):
        """
        Calculates power density [mW/mm^2] from arduino setting [fraction from 0 to 1]
        """
        avg_power = self.average_power(mW_density)
        #print("Avg power: ", avg_power)
        percent_intensity = self.power_to_arduino_setting(avg_power)
        return percent_intensity



    # Arduino Specific ---------------------------------------


    def init_arduino(self, arduino_path):
        """
        Initialize serial communication with arduino
        """

        #Helper function
        def open_link(path, baud):
            """Insert an object into pySerialtxfer TX buffer starting at the specified index.

            Args:
            txfer_obj: txfer - Transfer class instance to communicate over serial
            val: value to be inserted into TX buffer
            format_string: string used with struct.pack to pack the val
            object_byte_size: integer number of bytes of the object to pack
            start_pos: index of the last byte of the float in the TX buffer + 1

            Returns:
            start_pos for next object
            """
            link = txfer.SerialTransfer(path, baud) #'/dev/cu.usbmodem12401', baud=115200)
            link.open()
            time.sleep(2) # allow some time for the Arduino to completely resets
            return link


        self.arduino_path = arduino_path #'ASRL/dev/cu.usbmodem112401::INSTR'
        self.link = open_link(self.arduino_path, self.baud)



    def close_arduino(self, arduino_path):
            self.link.close()
            return


    def set_arduino_ramp(self, ramp = False, ramp_time = 1000):
            if not (0 <= ramp_time <= 65536): #
                raise ValueError("Arduino expects ramp_time to be 0 to max 2^16 (65536 ms)")
            self.ramp = ramp #ramp up signal 0 to max (over ramp_time duration) if True, square pulse if False
            self.ramp_time = ramp_time #ms, 1000ms = 1sec;
            if self.verbose: print("ramp:", self.ramp, "\nramp_time:", self.ramp_time, "ms")
            return


    #Helper function
    def send_datum(self, link, sent, format_string_send, format_string_rec):

        #Helper function --------------------------
        def stuff_object(txfer_obj, val, format_string, object_byte_size, start_pos=0):
            """Insert an object into pySerialtxfer TX buffer starting at the specified index.

            Args:
            txfer_obj: txfer - Transfer class instance to communicate over serial
            val: value to be inserted into TX buffer
            format_string: string used with struct.pack to pack the val
            object_byte_size: integer number of bytes of the object to pack
            start_pos: index of the last byte of the float in the TX buffer + 1

            Returns:
            start_pos for next object
            """
            val_bytes = struct.pack(format_string, *val)
            for index in range(object_byte_size):
                txfer_obj.txBuff[index + start_pos] = val_bytes[index]

            return object_byte_size + start_pos
        #Helper function --------------------------

        #----
        format_size = struct.calcsize(format_string_send)
        stuff_object(link, sent, format_string_send, format_size, start_pos=0)
        link.send(format_size)

        #----
        start_time = time.time()
        elapsed_time = 0
        while not link.available() and elapsed_time < 2:
            if link.status < 0:
                print('ERROR: {}'.format(link.status))
            else:
                if self.verbose: print('.', end='')
            elapsed_time = time.time()-start_time

        response =  link.rxBuff[:link.bytesRead]
        #print(response)

        binary_str = bytearray(response)
        #print(binary_str)
        result = struct.unpack(format_string_rec, binary_str)

        if self.verbose:
            print('SENT: %s' % str(sent))
            print('RCVD: %s' % str(result))
            print(' ')


    def send_arduino_datum(self, DAC_intensity_bitvalue, pulse, initial_delay, on_duration, off_duration):
        """
        Send data to arduino
        """
        data = []
        data.insert(0, DAC_intensity_bitvalue)
        data.append(self.ramp_time)
        data.append(self.ramp)
        data.append(self.use_maxwell)
        data.append(self.arduino_reply)
        data.append(pulse)
        data.append(initial_delay)
        data.append(on_duration)
        data.append(off_duration)

        sent = tuple(data)

        format_string_send = "HH????HHH" #"HH???" #'H64H?'#64h'
        format_string_rec = "HH????HHH" # "HH???"" #'H64H?'

        self.send_datum(self.link, sent, format_string_send, format_string_rec)
        return



    def set_arduino_intensity(self, intensity_fraction):

        self.arduino_intensity = intensity_fraction
        DAC_intensity_bitvalue = self.get_DAC_bitvalue(self.arduino_intensity)

        if self.verbose: print(DAC_intensity_bitvalue)
        if self.verbose: print("Expected Voltage:", 5*self.arduino_intensity)

        pulse = False
        self.send_arduino_datum(DAC_intensity_bitvalue, pulse, 0, 0, 0)

        return


    def arduino_pulse(self, intensity_fraction, initial_delay, on_duration, off_duration):
        """
        Send a pulse to the arduino
        """
        self.arduino_intensity = intensity_fraction
        DAC_intensity_bitvalue = self.get_DAC_bitvalue(self.arduino_intensity)

        pulse = True
        self.send_arduino_datum(DAC_intensity_bitvalue, pulse, initial_delay, on_duration, off_duration)

        return









'''
Notes:
- When LED Driver MOD receives >5V, it turns off
- DAC Documentation: https://www.sparkfun.com/datasheets/BreakoutBoards/MCP4725.pdf

LED Driver Mode: MOD

Set Brightness: 0-5V -- corresponds to intensity mW/mm2

Record how far your fiber is from the sample.

'''

# Docstring example
'''
Calculates power density [mW/mm^2] from arduino setting [fraction from 0 to 1]

Args:
    self:
    mW_density: float representing mW density

Returns:
    percent_intensity: arduino setting (float) as a fraction from 0 to 1

Raises:
    KeyError: Raises an exception.
    '''


# Checks before experiment (manual mode):
#1) Connect MaxWell headstage via HDMI cable to HDMI breakout board
#2) Remove little red board and unplug black & white jumper wire from it
#3) move black jumper wire (coming out of LEDD1B) to Arduino pin "GND"
#4) move white jumper wire (coming out of LEDD1B) to Arduino pin "A1"
#5) Set little black switch on LEDD1B to be "Trig" (middle position)
#6) Make sure power knob on LEDD1B is in on position (Past the click)
