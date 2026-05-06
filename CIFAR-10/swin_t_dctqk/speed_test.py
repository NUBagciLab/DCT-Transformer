# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 19:57:45 2023

@author: Zephyr
"""

import torch
import numpy as np
import numpy

# import torchvision
import swin_transformer

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print('Device:', device)
    
    # model = torchvision.models.get_model('swin_b', num_classes=1000).to(device)
    model = swin_transformer.swin_t().to(device)
    
    dummy_input = torch.randn(16, 3, 224, 224, dtype=torch.float).to(device)
    
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)