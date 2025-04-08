def convert_dat_to_h5(path_to_sims_dir, path_to_h5_out, cutoff=42, verbose=True):

    # known problem simulations:
    # TP1:
    #   - Run_TP_dyn_all_lanth_wind1_all_md0.058850_vd0.113508_mw0.097196_vw0.061553


    import h5py, glob, os
    import numpy as np
    from datetime import datetime

    t_min = 0.125
    t_max = 38.055
    times_in_common = np.logspace(np.log10(t_min), np.log10(t_max), 67) # times that ALL spectra have in common (some go to later times)

    if cutoff > 38.055:
        print("Warning: Not all spectra go out to %g days! Setting to common max time of 38.055 days." % cutoff)
        cutoff = 38.055

    t_idx = np.argmin(np.abs(cutoff-times_in_common))

    print('Time index: ', t_idx)

    files = np.array(glob.glob(path_to_sims_dir+'/*spec*'))
    files.sort()
    
    h5py_dataset = h5py.File(path_to_h5_out, 'w')

    if verbose:
        print('Converting spectra .dat files from %s' % path_to_sims_dir)
        print('Outputting spectra .hdf5 file to %s' % path_to_h5_out)
        print('Using %d time iterations with a maximum time of %g' % (t_idx+1, cutoff))
    
    start = datetime.now()

    for idx in range(files.shape[0]):
        spec_data = np.loadtxt(files[idx])
        spec_data = spec_data[:(t_idx+1)*1024, :] # only save times we care about
        spec_data = spec_data[:, 2:] # only look at angular data from 2 bins and higher
 
        # some spectra will have 0 photon and 0 noise in relevant bins
        # mark those with very low flux and very large noise
        # for well behaved training
        try:
            spec_data = np.array(np.split(spec_data, t_idx+1))
            print(spec_data.shape)
        except ValueError:
            print(spec_data.shape)
            print('Deleting file due to bad array split: ', files[idx])
            #os.remove(files[idx])
            continue
        for theta_idx in range(spec_data.shape[-1]):
            for time_idx in range(spec_data.shape[0]):
                # if looking at the spectral data, set zeroes to spectrum's lowest flux value
                if theta_idx < 54:
                    #spec_data[np.where(spec_data[:, theta_idx] <= 0)[0]] = np.min(np.nonzero(spec_data[:, theta_idx]))
                    nonzeros = np.nonzero(spec_data[time_idx, :, theta_idx])
                    spec_data[time_idx, np.where(spec_data[time_idx, :, theta_idx] <= 0)[0], theta_idx] = np.min(spec_data[time_idx, nonzeros, theta_idx])
                # if looking at Monte Carlo noise data, set zero noise comparable to spectrum flux value
                else:
                    zeros = np.where(spec_data[time_idx, :, theta_idx] <= 0)[0]
                    spec_data[time_idx, zeros, theta_idx] = spec_data[time_idx, zeros, theta_idx-54]

        try:
            all_data = np.concatenate((all_data, spec_data[None, :]), axis=0)
        except ValueError:
            print(spec_data.shape)
            print('Deleting file due to bad array concatenation: ', files[idx])
            #os.remove(files[idx])
            continue
        except NameError:
            all_data = spec_data[None, :]
        if verbose: print('Concatenated %d out of %d spectra' % (idx+1, files.shape[0]))
        del spec_data

    
    h5py_dataset.create_dataset('spectra', data=all_data)
    h5py_dataset.close()

    #print('Total conversion from dat to hdf5 took %g minutes' % (round((datetime.now()-start).total_seconds()/60), 3))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert a directory of SuperNu spec.dat outputs to hdf5 format for faster data I/O.")
    parser.add_argument('path_to_sims_dir', type=str, help='Path to directory containing SuperNu spec.dat data files.')
    parser.add_argument('path_to_hdf5_out', type=str, help='Path to hdf5 file output.')
    parser.add_argument('--cutoff', type=float, default=38.055, help='Time at which to cut off spectra. Default is maximum value in common for all spectra.') 
    
    args = parser.parse_args()
    
    convert_dat_to_h5(args.path_to_sims_dir, args.path_to_hdf5_out, args.cutoff)
