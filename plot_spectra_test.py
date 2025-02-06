import numpy as np
import spectra_emulator as SE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

se = SE.SpecEmulator('TP1', verbose=True)
print('initialized')

se.load_data()
print('data loaded')

se.preprocess_data()
print('data preprocessed')

se.split_training_data(se.X, se.y, se.y_noise, test_frac=0.50)

se.append_input_parameter(se.times, 1)
print('appended time as input parameter')

se.append_input_parameter(se.angles, 2)
print('appended angle as input parameter')

idx = np.random.randint(se.y_train.shape[0])
plt.figure()
plt.plot(se.wavs, se.y_train[idx], c='k', lw=1)
plt.fill_between(se.wavs, se.y_train[idx]-se.y_noise_train[idx], se.y_train[idx]+se.y_noise_train[idx])
plt.xscale('log')
plt.savefig('test_spectrum_random.pdf')
