

from MaxwellGI.optogenetics_enviroment import OptoEnv



class OptoCtl:
    
    def __init__(self, log_file, calibration_file, arduino_path):
        # Explain stuff, TBD...
        log_folder= log_file[:log_file.rfind("/")]
        log_filename = log_file.split("/")[-1]
        opto_env = OptoEnv( calibration_file, log_folder, verbose = True, manual_mode = False)
        opto_env.init_arduino(arduino_path)
        opto_env.set_stim_log(log_filename)
        opto_env.open_stim_log()
        self.opto_env = opto_env
        
        
    def pulses(self, on_duration, off_duration, num_pulses, intensity):
        self.opto_env.opto.set_arduino_intensity( intensity )
        # TO Do put the converstion function in here
        for i in range(num_pulses):
            self.opto_env.stim_pulse( on_duration*20, off_duration*20 ) # converting from ms to frames per second

            
    def close(self):
        self.opto_env.close()










