import numpy as np
import random

def vary_groups(group, n_el, options2choose, idx=0, random_array=np.array([], dtype='int32')):
    """
    Group sections and materials of the elements
    that are in the same group.
    """
    random_array_provided = False
    if random_array.shape[0] > 0:
        random_array_provided = True
            
    if group.shape[0] > 1:
        arr = np.zeros((n_el))
        group_array = np.ones((group.shape[0]))
        
        for g in range(group.shape[0]):
            if random_array_provided:
                selected = options2choose[random_array[g]][idx]
            else:
                random_array = np.concatenate((random_array, [random.randint(0, options2choose.shape[0]-1)]))
                random_array = random_array.astype('int32')
                selected = options2choose[random_array[-1]][idx]

            group_array[g] = selected
            for element in group[g]:
                if element <= 0:
                    break
                arr[element-1] = selected
    else:
        if random_array_provided:
            selected = options2choose[random_array[-1]][idx]
        else:
            random_array = np.array([random.randint(0, options2choose.shape[0]-1)], dtype='int32')
            selected = options2choose[random_array[-1]][idx]
        group_array = np.array([selected])
        arr = np.copy(group_array)
        

    return group_array, arr, random_array


def group_results(Fs, group):
    """
    Group F and sigma of the elements
    that are in the same group.
    """
    if group.shape[0] > 1:
        group_array = np.zeros((group.shape[0]))
        for g in range(group.shape[0]):
            for element in group[g]:
                if element <= 0:
                    break
                if abs(Fs[element-1]) > abs(group_array[g]):
                    group_array[g] = Fs[element-1]
    else:
        group_array = np.array([np.max(np.abs(Fs))], dtype='float64')
    
    return group_array

def convert_time(s):
    """
    Provided a time 's' in seconds,
    this method converts to a string
    in a format hh:mm:ss
    """
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))
