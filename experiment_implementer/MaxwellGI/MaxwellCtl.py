import maxlab
import maxlab.chip
import maxlab.saving
import maxlab.util

class MaxwellCtl:
    array = None # object used to control the electrode array on the maxwell 
    saver = None # object used to control the data saving procedures on the maxwell
    
    def __init__( self, config, gain=512, spike_threshold=5):
        """Turn on Maxwell device, set parameters, and then create an array object to control electrodes"""
        
        maxlab.util.initialize()                                         # start up chip
        #maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))  # make stim experiments possible (move this later)
        maxlab.send(maxlab.chip.Amplifier().set_gain(gain))              # Set gain 
        maxlab.send_raw(f"stream_set_event_threshold {spike_threshold}") # Set spike threshold sensitivity
        
        self.array = maxlab.chip.Array()                 # Create object for controling electrode array
        self.array.reset()                               # Delete any previous array
        self.array.load_config(config)                   # Load recording electrodes
        print('MaxOne initialized')                      # Once finished, print confirmation to user
        #self.array.route()   # Is this needed?
        self.array.download()                            # Downloads current electrode configuration to chip
        maxlab.util.offset()
        
        
    def recordingStart(self, data_file, only_spikes=False):
        """Start a recording to file_path. 
        file_path - must be ful filepath, eg: /home/my_dir/my_file.h5
        only_spikes - Determines whether or not to record all raw data or only spikes raster 
        """

        self.saver = maxlab.saving.Saving()          # Set up file and wells for recording, 
        self.saver.open_directory( data_file.rsplit("/",1)[0] )       
        self.saver.set_legacy_format(True)
        self.saver.group_delete_all()

        if not only_spikes:             # start recording and save results
            self.saver.group_define(0, "routed")
        self.saver.start_file( data_file.split("/")[-1].split(".h5")[0] )
        #print("Recording Started")
        self.saver.start_recording( range(1) )
        print("Recording Started")
    
    
    def recordingStop(self):
        """Stop maxwell recording"""
        self.saver.stop_recording()
        self.saver.stop_file()
        self.saver.group_delete_all()
        print("Recording finished")
    


