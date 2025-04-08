import torch
import numpy as np
from scipy.integrate import simpson

# 4-D case
#Best trial config: {'l1': 256, 'l2': 128, 'l3': 32, 'pd': 0, 'act_func': <class 'torch.nn.modules.activation.Tanh'>, 'lr': 0.0038766837162584164, 'batch_size': 128}
#Best trial final validation loss: 0.025537945330142975

# 6-D case
#Best trial config: {'l1': 256, 'l2': 128, 'l3': 2048, 'pd': 0, 'act_func': <class 'torch.nn.modules.activation.Tanh'>, 'lr': 0.0003637471404512984, 'ba      tch_size': 32}
#Best trial final validation loss: 0.04097653446602113
#Best trial config: {'l1': 32, 'l2': 32, 'l3': 2048, 'pd': 0, 'act_func': <class 'torch.nn.modules.activation.SELU'>, 'lr': 0.0016261558114465573, 'batc      h_size': 64}
#Best trial final validation loss: 0.03935224911189667

class Regressor(torch.nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.model = torch.nn.Sequential(
        torch.nn.Linear(6, 32),
        #torch.nn.Linear(4, 256),
        torch.nn.SELU(),
        torch.nn.Dropout(p=0),
        torch.nn.Linear(32, 32),
        torch.nn.SELU(),
        torch.nn.Dropout(p=0),
        torch.nn.Linear(32, 2048),
        torch.nn.SELU(),
        torch.nn.Linear(2048, 384),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class SpecEmulator(object):

    def __init__(self, dataset, verbose=False, seed=False):

        # Scratch space location

        scratch = '/lustre/scratch4/turquoise/mristic'

        # Data-specific settings

        dataset_settings = {
            'TP1': {
                "path_to_sims_dir": '{}/spectra/TP1'.format(scratch),
                "path_to_hdf5_data": '{}/TP1_spec.h5'.format(scratch),
                "times": np.logspace(np.log10(0.125), np.log10(34.896), 66) # knsc1.2 spectra go out to 34.896 days

            }
        }

        for key, value in dataset_settings[dataset].items():
            setattr(self, key, value)

        self.wavs = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4
        self.angles = np.degrees([np.arccos((1 - (i-1)*2/54.)) for i in np.arange(1, 55, 1)])

        # PyTorch settings

        self.get_device()
        self.model = Regressor().to(self.device)
        self.loss = self.relative_squared_error_loss
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1.626e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optim, lr_lambda=0.995)

        self.batch_size = 64 # smaller batch size = better ability to generalize!
        self.validation_model_path = 'pytorch_models/regression_validation.pt'

        # Miscellaneous settings
      
        self.seed = seed if seed else np.random.randint(int(1e6)) 
        np.random.seed(self.seed)
        self.verbose = verbose

        if self.verbose: 'seed = ', print(self.seed)

        return

    def get_device(self):
        
        ''' Trains on CUDA GPUs if available, otherwise uses CPUs '''

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def split_training_data(self, X, y, test_frac=0.50):

        from sklearn.model_selection import train_test_split

        self.X_train, X_test, self.y_train, y_test = train_test_split(
            X, y, test_size=test_frac)

        # create validation set that is 50% of the test set size
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
        X_test, y_test, test_size=0.50)

        if self.verbose:
            print('Size of X_train: ', self.X_train.shape, '   Size of y_train: ', self.y_train.shape)
            print('Size of X_valid: ', self.X_valid.shape, '   Size of y_valid: ', self.y_valid.shape)
            print('Size of X_test:  ', self.X_test.shape,  '   Size of y_test:  ', self.y_test.shape)

    def load_data(self, trim=True):

        import glob, h5py

        def get_params_from_filename(string):
            name_parse = string.split('/')[-1].split('_')
            params = np.array([name_parse[7][2:], name_parse[8][2:], name_parse[9][2:], name_parse[10][2:]]).astype('float')
            return params

        sim_files = np.array(glob.glob(self.path_to_sims_dir+'/*spec*'))
        sim_files.sort()

        for idx in range(sim_files.shape[0]):
            param = get_params_from_filename(sim_files[idx]) # parse filenames into parameters
            try: self.X = np.concatenate((self.X, param[None, :]), axis=0)
            except AttributeError: self.X = param[None, :]

        h5_file = h5py.File(self.path_to_hdf5_data, 'r')
        self.y = h5_file['spectra'][:]

        if trim:
            self.t_idxs = np.where((self.times > 1.4) & (self.times < 10.4))[0] # AT2017gfo observations
            self.wav_idxs = np.where((self.wavs > 0.39) & (self.wavs < 2.4))[0] # between LSST g and 2MASS K bands
            self.angle_idxs = np.arange(len(self.angles)//2) # keep all angles for now, can reduce by factor of 2 if data volume too big

        else:
            self.t_idxs = np.arange(len(self.times))
            self.wav_idxs = np.arange(len(self.wavs))
            self.angle_idxs = np.arange(len(self.angles))
 
        self.times = self.times[self.t_idxs]
        self.wavs = self.wavs[self.wav_idxs]
        self.angles = self.angles[self.angle_idxs]
        
        # first split spectra and noise arrays
        self.y = self.y[:, :, :, self.angle_idxs]
        self.y = self.y[:, :, self.wav_idxs, :]
        self.y = self.y[:, self.t_idxs, :, :]

        #self.y = self.y[:, 10, :, 0]*54

        if self.verbose:
            print('Size of X: ', self.X.shape, ' Size of y: ', self.y.shape)
 
    def preprocess_data(self):
      
        # logarithm of mass inputs for logical dynamic range 
        #self.X_train[:, 0] = np.log10(self.X_train[:, 0])
        #self.X_train[:, 2] = np.log10(self.X_train[:, 2])
   
        #self.X_valid[:, 0] = np.log10(self.X_valid[:, 0])
        #self.X_valid[:, 2] = np.log10(self.X_valid[:, 2])
   
        #self.X_test[:, 0] = np.log10(self.X_test[:, 0])
        #self.X_test[:, 2] = np.log10(self.X_test[:, 2])

        self.X[:, 0] = np.log10(self.X[:, 0])
        self.X[:, 2] = np.log10(self.X[:, 2])

        # logarithm of spectra for easier training 
        #self.y_train = np.log10(self.y_train)
        #self.y_valid = np.log10(self.y_valid)
        #self.y_test = np.log10(self.y_test)

        self.y = np.log10(self.y)       

        self.X_mu, self.y_mu = np.mean(self.X, axis=0), np.mean(self.y)
        self.X_sigma, self.y_sigma = np.std(self.X, axis=0), np.std(self.y)

        #self.X_train = (self.X_train - self.X_mu)/self.X_sigma
        #self.X_valid = (self.X_valid - self.X_mu)/self.X_sigma
        #self.X_test = (self.X_test - self.X_mu)/self.X_sigma

        self.X = (self.X - self.X_mu)/self.X_sigma        

        #self.y_train = (self.y_train - self.y_mu)/self.y_sigma
        #self.y_valid = (self.y_valid - self.y_mu)/self.y_sigma
        #self.y_test = (self.y_test - self.y_mu)/self.y_sigma
    
        self.y = (self.y - self.y_mu)/self.y_sigma

        return

    def append_input_parameter(self, values_to_append, axis_to_append):

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

        self.X_train, self.y_train = df1d.format(self.X_train, self.y_train, values_to_append, axis_to_append)
        self.X_valid, self.y_valid = df1d.format(self.X_valid, self.y_valid, values_to_append, axis_to_append)
        self.X_test, self.y_test = df1d.format(self.X_test, self.y_test, values_to_append, axis_to_append)

        print(self.X_train.shape, self.y_train.shape)
        print(self.X_valid.shape, self.y_valid.shape)

        return  

    def relative_squared_error_loss(self, y, y_hat):
        numerator = (y-y_hat)**2
        denominator = (y - y.mean())**2
        
        alpha = 0.001
        L_bol_y_hat = simpson(self.to_numpy(y_hat), self.wavs)
        L_bol_y = simpson(self.to_numpy(y), self.wavs)
        L_bol_term = np.abs(L_bol_y_hat - L_bol_y).sum()
        L_bol_term = np.exp(-alpha*L_bol_term)
        
        return numerator.sum()/denominator.sum()*L_bol_term

    def reduced_chi2_loss(self, y_hat, y):
        #degree_of_freedom = y.shape[0] - self.X.shape[1]
        residual = (y-y_hat)**2
        # if angle included: 
        # L_bol_y_hat = simpson(y_hat.sum(axis=-1), self.wavs)
        # L_bol_y = simpson(y_hat.sum(axis=-1), self.wavs)
        #alpha = 0.01
        #L_bol_y_hat = simpson(self.to_numpy(y_hat), self.wavs)
        #L_bol_y = simpson(self.to_numpy(y), self.wavs)
        #L_bol_term = np.abs(L_bol_y_hat - L_bol_y).sum()
        #L_bol_term = np.exp(-alpha*L_bol_term)
        #return 1/y.shape[0]*residual.sum()*L_bol_term
        return 1/y.shape[0]*residual.sum()

    def smooth_data(self, x, N):
       
        ''' 
        Smoothes data to get general profile for earlier training epochs prior to introducing more detailed features

        x: array_like
            Array of data to smooth.

        N: integer
            Size of smoothing window.
        '''
 
        from scipy.ndimage import uniform_filter1d
        
        try: return torch.Tensor(uniform_filter1d(self.to_numpy(x), size=N)).to(self.device)
        except: return torch.Tensor(uniform_filter1d(x, size=N)).to(self.device)

    def train(self, epochs):
        
        from torch.utils.data import TensorDataset, DataLoader
        from copy import deepcopy
       
        self.X_train, self.X_valid, self.X_test = \
            torch.Tensor(self.X_train).to(self.device), \
            torch.Tensor(self.X_valid).to(self.device), \
            torch.Tensor(self.X_test).to(self.device)
        
        self.y_train, self.y_valid, self.y_test = \
            torch.Tensor(self.y_train).to(self.device), \
            torch.Tensor(self.y_valid).to(self.device), \
            torch.Tensor(self.y_test).to(self.device)

        dataset = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        lim_valid = 1e10
        #reduce_window_size = False

        for epoch in range(epochs):

            loss_train = 0
            self.model.train()

            smoothing_window_size = np.max([30-epoch, 5])

            # for curriculum learning, we introduce a smoothing window starting with size 250
            # the spectra start oversmoothed, and with each epoch, the size of the smoothing window is decreased
            # by the end, we train on the spectra with all the detailed features
            # purpose of this is to gently guide the NN to learning the fully-featured spectrum
            # by starting with simple, blackbody-like spectra

            #if reduce_window_size: 
            #    smoothing_window_size -= 1
            for step, (batch_X, batch_y) in enumerate(loader):
                batch_y_hat = self.model(batch_X)
                #if smoothing_window_size > 1: 
                #loss_batch = self.loss(batch_y_hat, self.smooth_data(batch_y, N=smoothing_window_size))
                #else:
                #    # once smoothing window size <= 1, just use the unsmoothed data
                loss_batch = self.loss(batch_y_hat, batch_y)

                self.optim.zero_grad()
                loss_batch.backward()
                self.optim.step()
        
                loss_train += loss_batch

            loss_train /= (step+1)

            with torch.no_grad():
                self.model.eval()
                loss_valid =  self.loss(self.model(self.X_valid), self.y_valid)

            #if (epoch+1) % 10 == 0: 
            print('Epoch: ', epoch+1, ' Training loss: ', loss_train.item(), ' Validation loss: ', loss_valid.item())#, ' Smoothing window size: ', smoothing_window_size)

            # if model improves on best validation error so far, save it!
            if loss_valid <= lim_valid:

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': deepcopy(self.model.state_dict()),
                    'optimizer_state_dict': deepcopy(self.optim.state_dict()),
                    'loss_valid': loss_valid,
                }, self.validation_model_path)
            
                lim_valid = loss_valid

            # every 10 epochs, reload the model with lowest validation loss to date
            if (epoch+1) % 10 == 0:
                print('Reloading best model so far...') 
                checkpoint = torch.load(self.validation_model_path, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
                lim_valid = checkpoint['loss_valid'] # only keep new models that beat the best so far
                print('Validation loss: ', lim_valid.item())#, ' Smoothing window size: ', smoothing_window_size)

    def save(self, model_path):

        from copy import deepcopy
        
        torch.save({
            'model_state_dict': deepcopy(self.model.state_dict()),
            'optimizer_state_dict': deepcopy(self.optim.state_dict()),
            'seed': self.seed
        }, model_path)

        return    

    def load(self, model_path, use_seed=True):
        
        checkpoint = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        if use_seed: np.random.seed(checkpoint['seed'])
        print(checkpoint['seed'])
    
        return

    def plot_random_valid(self):
        
        import os, shutil
        import matplotlib       
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        idx = np.random.randint(self.X_valid.shape[0], size=20)
        X_valid = torch.Tensor(self.X_valid[idx]).to(self.device)
        y_valid = self.y_valid[idx]
        y_hat = self.model(X_valid)
        
        # undo transformations
        y_valid = y_valid*self.y_sigma + self.y_mu
        y_valid = 10**y_valid

        y_hat = y_hat*self.y_sigma + self.y_mu
        y_hat = 10**y_hat

        shutil.rmtree('./figures', ignore_errors=True)
        os.mkdir('./figures')

        for i in range(y_hat.shape[0]):

            plt.figure()
            plt.plot(self.wavs, self.to_numpy(self.smooth_data(y_valid[i], 5)), c='k')
            plt.plot(self.wavs, self.to_numpy(self.smooth_data(y_hat[i], 5)), c='r')
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig('./figures/fit_%d.pdf' % i)
