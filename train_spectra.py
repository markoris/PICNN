import numpy as np
import spectra_emulator as SE

se = SE.SpecEmulator('TP1', verbose=True)
print('initialized')

se.load_data()
print('data loaded')

se.preprocess_data()
print('data preprocessed')

se.split_training_data(se.X, se.y, se.y_noise, test_frac=0.50)

#se.append_input_parameter(se.times, 1)
#print('appended time as input parameter')

#se.append_input_parameter(se.angles, 2)
#print('appended angle as input parameter')

se.get_device()
print('device: ', se.device)

se.train()

se.plot_random_valid()
