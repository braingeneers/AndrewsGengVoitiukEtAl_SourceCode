

import zmq
import struct
from collections import namedtuple

#import MaxwellGI.Controller


SpikeEvent = namedtuple('SpikeEvent', 'frame channel amplitude')
_spike_struct = '8xLif'
_spike_struct_size = struct.calcsize(_spike_struct)



class Streamer:
    """Creates environment for monitoring live stream of ephys data"""
    subscriber = None # Used to listen to raw data
    
    def __init__( self, maxwell_ctl, filtered=True):
        """Prepare to listen to data stream from Maxwell"""
        #? maxwell_ctl is currently not used, it's added just to force user to first initialize the maxwell
        # Connect python to to raw data stream from Maxwell
        self.subscriber = zmq.Context.instance().socket(zmq.SUB) # Used to listen to raw data
        self.subscriber.setsockopt(zmq.RCVHWM, 0)
        self.subscriber.setsockopt(zmq.RCVBUF, 10*20000*1030)
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self.subscriber.setsockopt(zmq.RCVTIMEO, 100)
        stream_name= 'tcp://localhost:7205' if filtered else 'tcp://localhost:7204'
        self.subscriber.connect( stream_name )
        
        # ignore any first few partial packets to make sure subscriber is aligned to data stream frames
        # maxlab.util.offset()
        more = True
        while more:
            try:
                _ = self.subscriber.recv()
            except zmq.ZMQError:
                if self.time_elapsed() >= 3:
                    raise TimeoutError("Make sure the Maxwell Server is on.")
                continue
            more = self.subscriber.getsockopt(zmq.RCVMORE)   
        
        print("maxwell streamer ready")    
        
        
        
    def getData(self):
        '''
        Use the subscriber to capture the frame and event data from the server.
        Returns an integer frame_number as well as data buffers for the voltage
        '''
        frame_number = frame_data = events_raw = None
        
        # Get Data from subscriber
        # Sometimes the publisher will be interrupted, so fail cleanly by terminating this run of the environment, returning done = True.
        try:
            frame_number = struct.unpack('Q', self.subscriber.recv())[0] # unpack first component of each message, the frame number, a long long

            if self.subscriber.getsockopt(zmq.RCVMORE):   # Get raw voltage data
                frame_data = self.subscriber.recv()
            if self.subscriber.getsockopt(zmq.RCVMORE):   # Store spikes data
                events_raw = self.subscriber.recv()

        except Exception as e:
            #if self.debug > 1:
            print(e)

        # Reformat events data, see ash's _parse_events_list
        events = [] # Parse the raw binary events data into a list of SpikeEvent objects.
        if events_raw is not None:
            # The spike structure is 8 bytes of padding, a long frame number, an integer channel (the amplifier, not the
            # electrode), and a float amplitude.
            if len(events_raw) % _spike_struct_size != 0:
                print(f'Events has {len(events_raw)} bytes,', f'not divisible by {_spike_struct_size}', file=sys.stderr)

            # Iterate over consecutive slices of the raw events data and unpack each one into a new struct.
            for i in range(0, len(events_raw), _spike_struct_size):
                ev = SpikeEvent(*struct.unpack(_spike_struct, events_raw[i:i+_spike_struct_size]))
                events.append(ev)
            
        return frame_number, frame_data, events        
        






