# -*- coding: utf-8 -*-

import cython
import numpy as np
cimport numpy as np


cpdef inline double frate_total(double v_fb):
    """Calculate the feedback voltage dependet totel tunneling rate from the given polynom.
    
    v_fb: feedback voltage
    """
    return 180000 * v_fb + 2500

@cython.cdivision(True)
cdef inline double feedback(double v_fb,
                            double v_fb_max, 
                            double v_fb_min,
                            double fb_factor, 
                            double fb_time, 
                            double n_t,
                            long long n_i):
    """Feedback loop implementation with boundarie check
    
    v_fb: feedback start voltage
    v_fb_max: maximum feedback voltage
    v_fb_min: minimum feedback voltage
    v_fb: feedback factor
    v_time: feedback time window
    n_t: feedback target events
    n_i: feedback window events
    """
    
    v_fb_next = v_fb + fb_factor / fb_time * (n_t - n_i)

    # Boundary check
    if v_fb_next > v_fb_max:
        v_fb_next = v_fb_max
    elif v_fb_next < v_fb_min:
        v_fb_next = v_fb_min
                
    return v_fb_next


@cython.cdivision(True)
def feedback_generator_barrier(long long nr_of_events,
                               double v_fb,
                               double v_fb_max,
                               double v_fb_min,
                               double fb_targetrate,
                               double fb_factor, 
                               double fb_time,
                               double sampling_rate=400e3):
    """Generate a feedback signal.
    
    nr_of_events: generate a signal with so many events
    v_fb: feedback start voltage
    v_fb_max: maximum feedback voltage
    v_fb_min: minimum feedback voltage
    v_fb: target feedback rate
    v_fb: feedback factor
    v_time: feedback time window
    sampling_rate: number of samples per second.
    """
        
    # Devide by experimental voltage devider
    fb_factor = fb_factor / 31
    
    # Define internal counters
    cdef double probability = frate_total(v_fb) / sampling_rate 
    
    cdef long long state_length = 1  # Length of current state
        
    cdef long long n_i = 0   # Number of events in current feedback window
    cdef double n_t = fb_targetrate * fb_time
    
    cdef long long fb_window_position = 0
    cdef long long fb_window_length = <long long>(fb_time * sampling_rate)
    
    # Jump into generator loop
    while True:
        
        signal = list()
    
        # Run until the number of events happend
        for _ in range(nr_of_events):
            
            # Calculate the tunneling probability
            probability = frate_total(v_fb) / sampling_rate 
            state_length = 0
            
            # Roll the dice
            while True:
                change = probability >= np.random.random()
                
                state_length += 1
                fb_window_position +=1
                
                # roll the dice and check for the event
                
                
                # Check for feedback response
                if fb_window_position == fb_window_length:
                    
                    # Calculate feedback response
                    v_fb = feedback(v_fb, v_fb_max, v_fb_min, fb_factor, fb_time, n_t, n_i)
                    
                    # Calculate new tuneling probability
                    probability = frate_total(v_fb) / sampling_rate       
                    
                    # Set back window and feedback events
                    n_i = 0
                    fb_window_position = 0
                
                # Check for tunneling event
                if change:
                    break
            
            # Append current event to signal list
            signal.append((0, state_length, 0))
            
            n_i += 1
        
        # Return the signal and yield the generator
        yield signal