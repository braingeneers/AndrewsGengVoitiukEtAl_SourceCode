
import opto_hardware
from opto_hardware import OptoHardware

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches


import itertools
from itertools import tee                                         
import time
from datetime import datetime, timezone

import pytz
from pytz import timezone, utc

import csv
from csv import writer

import os.path


import re


# Import maxlab libraries if on MaxWell computer
# Skip using maxlab libraries if developing elsewhere
try:
    import maxlab
    import maxlab.system
    USE_MAXLAB = True
except ImportError:
    USE_MAXLAB = False
    print("Maxlab libraries not available on this computer, skipping usage")


#MaxWell LED COLORS
LIGHT_BLUE = 0
CYAN = 1
PINK = 2
DARK_BLUE = 3
LIGHT_GREEN = 4
DARK_GREEN = 5
RED = 6
NO_COLOR = 7


class OptoEnv:

    """ Opto Environment
    Environment for interfacing with optogenetics on maxwell chip.
    Assumes single fiber/LED setup.
    """

    def __init__(self, parameters = None, recordings_folder = None, verbose = True, manual_mode = False):

        self.manual_mode = False #If True, set intensity knob by hand using "TRIG" setting on Thorlabs LEDD1B
        self.verbose = verbose
        self.parameters = parameters
        self.recordings_folder = recordings_folder
        self.arduino_path = None

        # Logging
        self.experiment_name = ""
        self.recording_filename = "" #input("Please indicate data recording filename: ")
        self.stim_log_file = "" #"path_to_file"
        self.stim_log_file_handle = None
        self.stim_log_writer = None
        self.stim_log_reader = None

        # MaxWell hardware parameters
        self.sec = 20000 #frames maxwell_second_fps
        self.ms = 20 #frames (1 frame is 50us)
        self.frame = 0.00005 #seconds per frame, (20,000 frames in 1 sec), 1/20000
        self.frame_us = 50 #microseconds per frame

        self.maxwell_max_delay_buffer = 65536 - 1 #2^16, ~3.2768 sec @ 20kHz sampling

        # Stim parameters
        self.maxwell_sequence_initial_delay = 0 #maxwell reccomens 500 samples (5ms) delay to avoid inter-sequence jitter
        self.on_duration = 10000 # MaxWell samples
        self.off_duration = 20000 # MaxWell sampels (20,000 == 1 sec?)
        # timing resolution accurate to 1 x 10^7 (0.0000001) = 0.1 us
        self.time_past = time.time() #was 0
        self.time_curr = time.time()
        self.time_stim_end = 0
        #self.opto.arduino_intensity


        # Start hardware enviroment
        self.opto = OptoHardware(verbose = False)

        # Load opto hardware parameters
        self.opto.set_path_to_parameter_csv(self.parameters)
        self.opto.load_parameter_csv()

        #List available USB devices for user to locate Arduino
        self.opto.list_USB_devices()
        #self.arduino_path = input("Please indicate arduino path (i.e. '/dev/cu.usbmodem112401'): ")
        #self.opto.init_arduino(self.arduino_path)

        # Set all 8 GPIO pins to outputs
        if USE_MAXLAB:
            maxlab.send( maxlab.system.GPIODirection(0b11111111))
            maxlab.send( maxlab.system.StatusLED(color=2))
            # Set GPIO pins low (start in no stim state)
            maxlab.send( maxlab.system.GPIOOutput(0b00000000))

        return


    def print_maxwell_colors(self):
        colors = "LIGHT_BLUE, CYAN, PINK, DARK_BLUE, LIGHT_GREEN, DARK_GREEN, RED, NO_COLOR"
        print(colors)


    def set_maxwell_color(self, color):
        if USE_MAXLAB: maxlab.send(maxlab.system.StatusLED(color))


    def ms_to_frames(self, ms):
        return self.ms * ms

    def sec_to_frames(self, sec):
        return self.sec * sec

    def frames_to_ms(self, frames):
        return frames / self.ms

    def frames_to_sec(self, frames):
        return frames / self.sec


    def init_arduino(self, arduino_path):
        self.arduino_path = arduino_path #input("Please indicate arduino path (i.e. '/dev/cu.usbmodem112401'): ")
        self.opto.init_arduino(self.arduino_path)
        return

    def set_stim_log(self, recording_filename):
        self.recording_filename = re.sub('\_opto_stim_log.csv$', '', recording_filename)
        self.stim_log_file = self.recording_filename + "_opto_stim_log.csv"
        self.stim_log_file = os.path.join(self.recordings_folder, self.stim_log_file)

        #if file doesn't exist, make it
        if not os.path.isfile(self.stim_log_file):
            self.recording_filename = (datetime.now(tz=pytz.timezone('US/Pacific')).strftime('%Y%m%dT%H%M%S-')) + self.recording_filename #input("Please indicate data recording filename: ")
            self.stim_log_file = self.recording_filename + "_opto_stim_log.csv"
            self.stim_log_file = os.path.join(self.recordings_folder, self.stim_log_file)

        if not os.path.isdir(self.recordings_folder):
            os.mkdir(self.recordings_folder)


    def open_stim_log(self):
        if self.stim_log_file is None:
            raise ValueError("No stim_log_file! Use set_stim_log() to set the file.")

        print("recording_filename", self.recording_filename)
        print("stim_log_file:", self.stim_log_file)



        self.stim_log_file_handle = open(self.stim_log_file, 'a+', newline='')
        self.stim_log_writer = writer(self.stim_log_file_handle)
        self.stim_log_reader = csv.reader(self.stim_log_file_handle)

        self.stim_log_file_handle.seek(0)
        file_len = len(list(self.stim_log_reader))
        print("CSV file is", file_len)
        if file_len <= 0:  # write first row:
                self.stim_log_writer.writerow(['time (sec)', 'intensity_fraction', 'initial_delay (frames)', 'on_duration (frames)', 'off_duration (frames)', 'notes', "use_maxwell", "delta_t (sec)"])

        return


    def close_stim_log(self):
        if self.stim_log_file_handle is not None:
            self.stim_log_file_handle.close()


    def view_current_stim_log(self):
        self.stim_log_file_handle.seek(0)
        for row in self.stim_log_reader:
            print(row)

    def pairs(self, iterable):
        # Staggered pairs of s[n] and s[n+1]: 
        # (s[0],s[1]), (s[1],s[2]), ... (s[n],None) 
        iterable.append(None)
        a, b = tee(iterable) 
        next(b, None) 
        return zip(a, b) 


    def parse_log_to_plottable(self, slicer = None, time_scale = "sec"):

        if slicer == None: 
            slicer = slice(0, None)
       
       
        if time_scale == "sec":
            time_scale = self.sec
            #if self.verbose: print("in parse_log_plottable: time_scale is sec")
        elif time_scale == "ms":
            time_scale = self.ms
            #if self.verbose: print("in parse_log_plottable: time_scale is ms")
        else:
            raise ValueError("time_scale must be 'sec' or 'ms'")

        self.stim_log_file_handle.seek(0)
        self.stim_log_reader = csv.reader(self.stim_log_file_handle)
        data = list(self.stim_log_reader)[1:]

 
        for row, next_row in self.pairs(data[slicer]): # pairwise(b) also works 
            #print(row, next_row)

            time = float(row[0]) - float(data[0][0])
            #print("Time:", time, "\t row[0]:", row[0], "\t data[1][0]:", data[1][0])

            if time_scale == self.ms:
                time = time * 1000

            intensity_fraction = float(row[1])
            initial_delay = float(row[2])/time_scale
            use_maxwell = row[6].lower() in ("true", "t", "1")
            print("USE MAXwell:", use_maxwell, row[6])


            notes = row[5]

            if notes == "Single Command ON":
                if next_row != None:
                    on_duration = float(next_row[0]) - float(row[0]) #lookeahead int(row[3])/time_scale
                    off_duration = 0
            elif notes == "Single Command OFF":
                if next_row != None:
                    on_duration = float(next_row[0]) - float(row[0]) #lookeahead int(row[3])/time_scale
                    off_duration = 0
            else:
                on_duration = int(row[3])/time_scale
                off_duration = int(row[4])/time_scale

            t = time+initial_delay


            #plt.axvspan(t, t+on_duration, color=self.LED_wavelength_color, alpha=0.2)
            if self.verbose:
                print("Row:",row)
                print("Time:", time)
                print("Plotting params :", t, intensity_fraction, initial_delay, on_duration, off_duration)
                print()
            #yield (stim_start, stim_stop, intensity_fraction)
            #if t != t + on_duration:
            yield (t, t+on_duration, intensity_fraction, use_maxwell)



    def plot_log(self, slicer = None, time_scale = "sec"):
        title = "test"
        xlabel = time_scale

        plt.rcParams.update({'font.size': 12})
        plt.title(self.stim_log_file)
        plt.xlabel(xlabel, fontsize=12)
        plt.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = True, bottom = True)
        #plt.box(True) #remove box

        # plot stimulations
        for stim_pulse in self.parse_log_to_plottable(slicer, time_scale):
            (stim_start, stim_stop, intensity_fraction, use_maxwell) = stim_pulse
            plt.axvspan(stim_start, stim_stop, color=self.opto.LED_wavelength_color, alpha=intensity_fraction)
            if use_maxwell is False: 
                plt.axvspan(stim_start, stim_stop, 0, 0.1, color='black', alpha=intensity_fraction)

        #legeng
        #ax.legend((rects1[0]), ('use_maxwell = False'), handlelength=0.7)
        l1 = mpatches.Patch(color='black', label='use_maxwell = False')
        plt.legend(handles=[l1], handlelength=0.7, bbox_to_anchor=(0.28,-0.7), loc="upper left")

        # colorbar
        cmap = LinearSegmentedColormap.from_list(None, colors =[(1, 1, 1), self.opto.LED_wavelength_color]) #N=100
        cbar = plt.colorbar(ScalarMappable(cmap=cmap), ticks=[0, 1], pad=0.2, orientation='horizontal')
        cbar.ax.set_xticklabels(["0", str(round(self.opto.arduino_setting_to_power_density(1), 1))], fontsize=10)
        cbar.ax.set_xlabel("$mW/mm^2$", labelpad=-15, fontsize=10)
        #cbar.outline.set_visible(False)
        plt.tight_layout()

        plt.show()
        return



    def update_time(self, initial_delay, on_duration, off_duration):
        self.time_past = self.time_curr
        self.time_curr = time.time()
        self.time_stim_end = self.time_curr + self.frames_to_sec(initial_delay + on_duration + off_duration)



    def log_stim(self, intensity, initial_delay, on_duration, off_duration, notes):
        if self.stim_log_file is None:
            raise ValueError("No stim_log_file! Use set_stim_log() to set the file.")

        #print("Notes:", notes)
        if on_duration == 0 and not (notes == "Single Command ON" or notes == "Single Command OFF"):
            #print("RETURNING!")
            return

        #maxlab.saving.save_stim_log(self.stim_log_file, self.stim, self.stim_units)

        self.stim_log_writer.writerow([self.time_curr, intensity, int(initial_delay), int(on_duration), int(off_duration), notes, self.opto.use_maxwell, self.time_curr - self.time_past])

        if self.verbose:
            if notes == "Single Command ON":
                print("Stim toggled ON \t use_maxwell:", self.opto.use_maxwell, "\t arduino_intensity:", self.opto.arduino_intensity)
            elif notes == "Single Command OFF":
                print("Stim toggled OFF \t use_maxwell:", self.opto.use_maxwell, "\t arduino_intensity:", self.opto.arduino_intensity)
            else:
                print("Stim pulse \t use_maxwell:", self.opto.use_maxwell, "\t arduino_intensity:", self.opto.arduino_intensity, "\t delay/on/off (frames):", initial_delay, "/", on_duration, "/", off_duration)

        return


    def stim_on(self):
        self.update_time(0, 0, 0)
        if USE_MAXLAB and self.opto.use_maxwell: maxlab.send( maxlab.system.GPIOOutput(0b11111111))
        if self.opto.use_maxwell is False: self.opto.arduino_intensity = 0
        self.log_stim(self.opto.arduino_intensity, initial_delay=0, on_duration=0, off_duration=0, notes="Single Command ON")
        return


    def stim_off(self):
        self.update_time(0, 0, 0)
        if USE_MAXLAB and self.opto.use_maxwell: maxlab.send( maxlab.system.GPIOOutput(0b00000000))
        if self.opto.use_maxwell is False: self.opto.arduino_intensity = 0
        self.log_stim(intensity=0, initial_delay=0, on_duration=0, off_duration=0, notes="Single Command OFF")
        return


    def stitch_delay(self, duration):
        while(duration >= self.maxwell_max_delay_buffer):
            yield self.maxwell_max_delay_buffer
            duration = duration - self.maxwell_max_delay_buffer
        if duration > 0: yield duration


    def stim_pulse(self, on_duration, off_duration, notes=None): #consider: off_duration_pre, off_duration_post

        # Consider later for ramped signal & rampTime
        # if not (0 <= intensity_fraction <= 1):
        #         raise ValueError("rampTime")

        # if self.time_stim_end > time.time():
        #     print("Wait! Previous stim hasn't finished yet.")
        #     print("Stim Ends:", self.time_stim_end, "\t Time now:", time.time())

        while(self.time_stim_end > time.time()): pass

        initial_delay=self.maxwell_sequence_initial_delay


        if USE_MAXLAB and self.opto.use_maxwell:
            s = maxlab.Sequence(initial_delay=self.maxwell_sequence_initial_delay)

            s.append( maxlab.system.GPIOOutput(0b11111111))
            for on_duration_segment in self.stitch_delay(on_duration):
                    s.append( maxlab.system.DelaySamples(on_duration_segment))

            s.append( maxlab.system.GPIOOutput(0b00000000))
            for off_duration_segment in self.stitch_delay(off_duration):
                s.append( maxlab.system.DelaySamples(off_duration_segment))
        elif self.opto.use_maxwell is False: 
            self.opto.arduino_pulse(self.opto.arduino_intensity, initial_delay, on_duration, off_duration)
        #     self.opto.set_arduino_intensity(0)

        self.update_time(initial_delay, on_duration, off_duration)
        if USE_MAXLAB: s.send()
        self.log_stim(self.opto.arduino_intensity, initial_delay, on_duration, off_duration, notes)

        return



    def stim_sweep(self, intensities, on_durations, off_durations, num_stims_per_condition, off_between_conditions=0):
    # """
    # """
        for intensity in intensities:
            self.opto.set_arduino_intensity(intensity)
            for on_duration, off_duration in zip(on_durations, off_durations):
                for stim in range(num_stims_per_condition):
                    self.stim_pulse(on_duration, off_duration)
                self.stim_pulse(0, off_between_conditions)
        return


    # how much light intensity does it take to excite a neuron?
    def stim_intensity_sweep(self, intensities, on_duration, off_duration, num_stims_per_intensity, off_between_conditions=0):
        self.stim_sweep(intensities, [on_duration], [off_duration], num_stims_per_intensity, off_between_conditions)
        return

    # how much light exposure duration does it take to excite a neuron?
    def stim_duration_sweep(self, intensity, on_durations, off_durations, num_stims_per_duration, off_between_conditions=0):
        self.stim_sweep([intensity], on_durations, off_durations, num_stims_per_duration, off_between_conditions)    
        return

    # def stim_looping_pattern(self, intensities, on_durations, off_durations, num_stims_per_duration, off_between_conditions=0):
    #     self.stim_sweep(intensities, [on_duration], [off_duration], num_stims_per_intensity, off_between_conditions)
    #     return

    def freq_to_cycle_frames(self, frequencies):
       return [int(self.sec / frequency)  for frequency in frequencies]

    def freq_to_cycle_sec(self, frequencies):
        return [(1 / frequency)  for frequency in frequencies]

    #----------------------------------------------
 
    def dutycycle_to_duration_frames(self, frequency, dutycycle):
        cycle_duration = self.freq_to_cycle_frames([frequency])[0]

        on_duration = int(cycle_duration * dutycycle)
        off_duration = cycle_duration - on_duration
        
        return (on_duration, off_duration)


    def dutycycle_to_duration_sec(self, frequency, dutycycle):
        cycle_duration = self.freq_to_cycle_sec([frequency])[0]
        on_duration = cycle_duration * dutycycle
        off_duration = cycle_duration - on_duration

        return (on_duration, off_duration)


    # how much frequency does it take to excite a neuron?
    def stim_frequency_sweep_num(self, intensity, on_duration, frequencies, num_stims_per_frequency, off_between_conditions=0):
        cycle_durations = self.freq_to_cycle_frames(frequencies)

        if on_duration >= min(cycle_durations): raise ValueError(on_duration, ">", min(cycle_durations), "on_duration > min(cycle_durations). Please make on_duration smaller, or decesary the freuqency  (increase cycle)")

        # Note: frequency = 1 / (on_duration + off_duration)
        off_durations = [cycle_duration - on_duration for cycle_duration in cycle_durations]

        print("Frequencies:", frequencies)
        print("On durations", on_duration, "Off durations:", off_durations)
        self.stim_sweep([intensity], [on_duration]*len(off_durations), off_durations, num_stims_per_frequency, off_between_conditions)
        return


    #how much frequency does it take to excite a neuron?
    def stim_frequency_sweep_timed(self, intensity, on_duration, frequencies, time_sec_per_frequency, off_between_conditions):

        cycle_durations = self.freq_to_cycle_frames(frequencies)

        if on_duration >= min(cycle_durations): raise ValueError(on_duration, ">", min(cycle_durations), "on_duration > min(cycle_durations). Please make on_duration smaller, or decesary the freuqency  (increase cycle)")

        # Note: frequency = 1 / (on_duration + off_duration)
        off_durations = [cycle_duration - on_duration for cycle_duration in cycle_durations]

        print("Frequencies:", frequencies)
        print("On durations", on_duration, "Off durations:", off_durations)
        #----------------------

        self.opto.set_arduino_intensity(intensity)

        stim_count = 0
        for off_duration, frequency in zip(off_durations, frequencies):

            # if use_timer is True:

            for timer_sec in time_sec_per_frequency:
                if self.verbose: print("Stim " + str(timer_sec) + "sec (" + str(round(timer_sec/60, 2)) + \
                            " min) @ " + str(frequency) + " Hz  -(Stims every " + str(1/frequency) + " sec)")
        

                t_start = time.time()
                if self.verbose: print("Start: " + str(time.ctime(t_start)))
            
                t_end = time.time() + timer_sec

                while time.time() < t_end:
                    if self.verbose: print("Stim number: "+ str(stim_count))
                    stim_count+=1
                    self.stim_pulse(on_duration, off_duration)
            

                t_curr = time.time()
                if self.verbose: print("End: " + str(time.ctime(t_curr)))
                if self.verbose: print("total = " +  str(round(t_curr-t_start, 2)) + " sec")

            self.stim_pulse(0, off_between_conditions)

        return    


    def close(self):
        if self.manual_mode is False:
            self.opto.close_arduino(self.arduino_path)
        self.stim_log_file_handle.close()
        self.set_maxwell_color(0)
