from functools import partial
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ray import tune
from ray import train
from ray.tune import RunConfig, TuneConfig
from ray.train import Checkpoint, get_checkpoint, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import numpy as np
from scipy.integrate import simpson
from scipy.ndimage import uniform_filter1d

# Ray Tune settings

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
# Miscellaneous settings

scratch = '/lustre/scratch4/turquoise/mristic'
verbose = True

# Data-specific settings

dataset_settings = {
    'TP1': {
        "path_to_sims_dir": '{}/spectra/TP1'.format(scratch),
        "path_to_hdf5_data": '{}/TP1_spec.h5'.format(scratch),
        "times": np.logspace(np.log10(0.125), np.log10(34.896), 66) # knsc1.2 spectra go out to 34.896 days

    }
}

class Regressor(torch.nn.Module):
    def __init__(self, l1=128, l2=256, l3=512, pd=0.5, act_func=torch.nn.LeakyReLU):
        super(Regressor, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(6, l1),
            act_func(),
            torch.nn.Dropout(p=pd),
            torch.nn.Linear(l1, l2),
            act_func(),
            torch.nn.Dropout(p=pd),
            torch.nn.Linear(l2, l3),
            act_func(),
            torch.nn.Linear(l3, 384),
            )

    def forward(self, x):
        x = self.model(x)
        return x

def load_data(dataset_dict, trim=True):

        import glob, h5py

        def get_params_from_filename(string):
            name_parse = string.split('/')[-1].split('_')
            params = np.array([name_parse[7][2:], name_parse[8][2:], name_parse[9][2:], name_parse[10][2:]]).astype('float')
            return params

        sim_files = np.array(glob.glob(dataset_dict['path_to_sims_dir']+'/*spec*'))
        sim_files.sort()

        times = dataset_dict['times'] 
        wavs = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4
        angles = np.degrees([np.arccos((1 - (i-1)*2/54.)) for i in np.arange(1, 55, 1)])

        for idx in range(sim_files.shape[0]):
            param = get_params_from_filename(sim_files[idx]) # parse filenames into parameters
            try: X = np.concatenate((X, param[None, :]), axis=0)
            except: X = param[None, :]

        h5_file = h5py.File(dataset_dict['path_to_hdf5_data'], 'r')
        y = h5_file['spectra'][:]

        if trim:
            t_idxs = np.where((times > 1.4) & (times < 10.4))[0] # AT2017gfo observations
            wav_idxs = np.where((wavs > 0.39) & (wavs < 2.4))[0] # between LSST g and 2MASS K bands
            angle_idxs = np.arange(len(angles)//2) # keep all angles for now, can reduce by factor of 2 if data volume too big

        else:
            t_idxs = np.arange(len(times))
            wav_idxs = np.arange(len(wavs))
            angle_idxs = np.arange(len(angles))
 
        times = times[t_idxs]
        wavs = wavs[wav_idxs]
        angles = angles[angle_idxs]
        
        # first split spectra and noise arrays
        y = y[:, :, :, angle_idxs]
        y = y[:, :, wav_idxs, :]
        y = y[:, t_idxs, :, :]

        #y = y[:, 10, :, 0]*54

        if verbose:
            print('Size of X: ', X.shape, ' Size of y: ', y.shape)

        return X, y, times, angles

def split_training_data(X, y, test_frac=0.50):

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_frac)

        # create validation set that is 50% of the test set size
        X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=0.50)

        if verbose:
            print('Size of X_train: ', X_train.shape, '   Size of y_train: ', y_train.shape)
            print('Size of X_valid: ', X_valid.shape, '   Size of y_valid: ', y_valid.shape)
            print('Size of X_test:  ', X_test.shape,  '   Size of y_test:  ', y_test.shape)

        return X_train, X_valid, X_test, y_train, y_valid, y_test

def relative_squared_error_loss(y, y_hat):
        numerator = (y-y_hat)**2
        denominator = (y - y.mean())**2
        return numerator.sum()/denominator.sum()

def smooth_data(x, N):
 
        return torch.Tensor(uniform_filter1d(to_numpy(x), size=N)).to(device)
    
def to_numpy(x):
        return x.detach().cpu().numpy()

def transform(X, y):
    
    # normalize values between 0 and 1

    X_mu, X_sigma = X.mean(axis=0), X.std(axis=0)
    y_mu, y_sigma = y.mean(), y.std()

    X = (X - X_mu)/X_sigma
    y = (y - y_mu)/y_sigma

    return X, y, np.c_[X_mu, X_sigma], np.c_[y_mu, y_sigma]

def untransform(X, y, X_dist, y_dist):

    X = X*X_dist[1] + X_dist[0]
    y = y*y_dist[1] + y_dist[0]

    return X, y
    
def append_input_parameter(X_train, X_valid, X_test, y_train, y_valid, y_test, values_to_append, axis_to_append):

        '''
        Reduces the target array by 1 dimension to treat the replaced dimension as an input training variable.
        Spectra by default have shape [N, time, wavelength, angle] where N is the number of simulations.
        To add time as a training parameter, set t_max=None and angle=0 in load_data(), for example.
        This yields self.spectra.shape = [N, time, wavelength].
        The values_to_append array will be the times corresponding to the time column (axis=1) of self.spectra.
        Thus, this function would be called as append_input_parameter(self, self.times, 1).
        This would then yield self.params = [Md, vd, Mw, vd, t], and self.spectra would have shape [N*time, wavelength].
        Therefore, the dimension of self.spectra is reduced by 1, and the dimension of self.params is increased by 1.
        '''

        import data_format_1d as df1d

        X_train, y_train = df1d.format(X_train, y_train, values_to_append, axis_to_append)
        X_valid, y_valid = df1d.format(X_valid, y_valid, values_to_append, axis_to_append)
        X_test, y_test = df1d.format(X_test, y_test, values_to_append, axis_to_append)

        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)

        return  X_train, X_valid, X_test, y_train, y_valid, y_test

def train_regressor(config):
    net = Regressor(config["l1"], config["l2"], config["l3"], config["pd"], config["act_func"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = relative_squared_error_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=1e-4)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    X, y, times, angles = load_data(dataset_settings["TP1"])

    X[:, 0] = np.log10(X[:, 0])
    X[:, 2] = np.log10(X[:, 2])
 
    # logarithm of spectra for easier training 
    y = np.log10(y)
 
    # whiten the input X, output y, and output noise y_noise

    X, y, X_lims, y_lims = transform(X, y)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_training_data(X, y, test_frac=0.20)

    # append here

    X_train, X_valid, X_test, y_train, y_valid, y_test = append_input_parameter(X_train, X_valid, X_test, y_train, y_valid, y_test, times, 1) 
    X_train, X_valid, X_test, y_train, y_valid, y_test = append_input_parameter(X_train, X_valid, X_test, y_train, y_valid, y_test, angles, 2) 

    X_train, X_valid = \
        torch.Tensor(X_train).to(device), \
        torch.Tensor(X_valid).to(device), \
    
    y_train, y_valid = \
        torch.Tensor(y_train).to(device), \
        torch.Tensor(y_valid).to(device), \

    traindataset = TensorDataset(X_train, y_train)
    trainloader = DataLoader(traindataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
   
    valdataset = TensorDataset(X_valid, y_valid)
    valloader = DataLoader(valdataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    for epoch in range(start_epoch, 1000):  # loop over the dataset multiple times
    
        #smoothing_window_size = 200-5*epoch
        
        running_loss = 0.0
        epoch_steps = 0
        for i, (inputs, targets) in enumerate(trainloader):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            #if smoothing_window_size > 1: 
            #    loss = criterion(outputs, smooth_data(targets, N=smoothing_window_size))
            #else:
                # once smoothing window size <= 1, just use the unsmoothed data
            loss = criterion(targets, outputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.detach().cpu().numpy()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, (inputs, targets) in enumerate(valloader):
            with torch.no_grad():

                outputs = net(inputs)

                #if smoothing_window_size > 1: 
                #    loss = criterion(outputs, smooth_data(targets, N=smoothing_window_size))
                #else:
                    # once smoothing window size <= 1, just use the unsmoothed data
                loss = criterion(targets, outputs)
                val_loss += loss.detach().cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            #checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": running_loss / epoch_steps, "val loss": val_loss / val_steps},
                #checkpoint=checkpoint,
            )

    print("Finished Training")

def main(num_samples=100, max_num_epochs=10, gpus_per_trial=1):
    
    config = {
        "l1": tune.choice([2 ** i for i in np.arange(5, 12)]),
        "l2": tune.choice([2 ** i for i in np.arange(5, 12)]),
        "l3": tune.choice([2 ** i for i in np.arange(5, 12)]),
        "pd": tune.choice([0, 0.25, 0.50, 0.75]),
        "act_func": tune.choice([nn.LeakyReLU, nn.ReLU, nn.SELU, nn.Tanh]),
        "lr": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128, 256])
    }

    scheduler = ASHAScheduler(
        metric="val loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=20,
        reduction_factor=4,
    )

    result = tune.run(
        train_regressor,
        resources_per_trial={"cpu": 0.5, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path="/lustre/scratch4/turquoise/mristic/ray_results",
        verbose=1,
    )

#    regressor_with_resources = tune.with_resources(train_regressor, {"gpu": gpus_per_trial})
#    tuner = tune.Tuner(
#        regressor_with_resources,
#        param_space=config,
#        tune_config=TuneConfig(num_samples=num_samples, scheduler=scheduler),
#        run_config=RunConfig(storage_path="/lustre/scratch4/turquoise/mristic/ray_results", verbose=1),
##        scaling_config=ScalingConfig(accelerator_type="A100")
#    )
#
#    result = tuner.fit()

    best_trial = result.get_best_trial("val loss", "min", "all")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val loss']}")

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=5000, max_num_epochs=50, gpus_per_trial=0.5)

