import torch
import numpy as np

class Regressor(torch.nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.50),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.50),
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.50),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024)
            )

    def forward(self, x):
        x = self.model(x)
        return x

class SpecEmulator(object):

    def __init__(self, dataset, verbose=False):

        # Scratch space location

        scratch = '/lustre/scratch4/turquoise/mristic'

        # Data-specific settings

        dataset_settings = {
            'TP1': {
                "path_to_sims_dir": '{}/spectra/TP1'.format(scratch),
                "path_to_hdf5_data": '{}/TP_wind1_spec.h5'.format(scratch),
                "times": np.logspace(np.log10(0.125), np.log10(34.896), 66)
            }
        }

        for key, value in dataset_settings[dataset].items():
            setattr(self, key, value)

        self.wavs = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4
        self.angles = np.degrees([np.arccos((1 - (i-1)*2/54.)) for i in np.arange(1, 55, 1)])

        # PyTorch settings

        self.get_device()
        self.model = Regressor().to(self.device)
        self.loss = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.1, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optim, lr_lambda=0.995)

        self.epochs = 2000
        self.batch_size = 128
        self.validation_model_path = 'pytorch_models/regression_validation.pt'

        # Miscellaneous settings

        self.verbose = verbose

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

        self.X_train, self.X_valid, self.X_test = \
            torch.Tensor(self.X_train).to(self.device), \
            torch.Tensor(self.X_valid).to(self.device), \
            torch.Tensor(self.X_test).to(self.device)
        
        self.y_train, self.y_valid, self.y_test = \
            torch.Tensor(self.y_train).to(self.device), \
            torch.Tensor(self.y_valid).to(self.device), \
            torch.Tensor(self.y_test).to(self.device)

        if self.verbose:
            print('Size of X_train: ', self.X_train.shape, '   Size of y_train: ', self.y_train.shape)
            print('Size of X_valid: ', self.X_valid.shape, '   Size of y_valid: ', self.y_valid.shape)
            print('Size of X_test:  ', self.X_test.shape,  '   Size of y_test:  ', self.y_test.shape)

    def load_data(self):

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
        h5_data = h5_file['spectra'][:]
        self.y = h5_data.reshape(sim_files.shape[0], len(self.times), len(self.wavs), len(self.angles))

        # TEMPORARILY, UNTIL MORE FEATURES ADDED AS FIT PARAMETERS
        self.y = self.y[:, 10, :, 0] 

        if self.verbose:
            print('Size of X:       ', self.X.shape, '   Size of y: ', self.y.shape)
 
        # knsc1.2 spectra go out to 34.896 days

    def whiten_data(self):
        
        self.mu_X, self.mu_y = np.mean(self.X, axis=1), np.mean(self.y)
        self.sigma_X, self.sigma_y = np.std(self.X, axis=1), np.std(self.y)

        self.X = (self.X - self.mu_X)/self.sigma_X
        self.y = (self.y - self.mu_y)/self.sigma_y

        return

    def smooth_data(self, x, N):
       
        ''' 
        Smoothes data to get general profile for earlier training epochs prior to introducing more detailed features

        x: array_like
            Array of data to smooth.

        N: integer
            Size of smoothing window.
        '''
 
        from scipy.ndimage import uniform_filter1d
        
        return torch.Tensor(uniform_filter1d(self.to_numpy(x), size=N)).to(self.device)

    def train(self):
        
        from torch.utils.data import TensorDataset, DataLoader
        from copy import deepcopy

        dataset = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        lim_valid = 1e10

        for epoch in range(self.epochs):

            loss_train = 0

            smoothing_window_size = 250-epoch
            for step, (batch_X, batch_y) in enumerate(loader):

                batch_y_hat = self.model(batch_X)
                if smoothing_window_size > 1: 
                    loss_batch = self.loss(batch_y_hat, self.smooth_data(batch_y, N=smoothing_window_size))
                else:
                    # once smoothing window size <= 1, just use the unsmoothed data
                    loss_batch = self.loss(self.to_numpy(batch_y_hat), self.to_numpy(batch_y))

                self.optim.zero_grad()
                loss_batch.backward()
                self.optim.step()
        
                loss_train += loss_batch

            loss_train /= (step+1)

            loss_valid =  self.loss(self.model(self.X_valid), self.y_valid)

            print('Epoch: ', epoch, ' Training loss: ', loss_train.item(), ' Validation loss: ', loss_valid.item())

            # if model improves on best validation error so far, save it!
            if loss_valid <= lim_valid:

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': deepcopy(self.model.state_dict()),
                    'optimizer_state_dict': deepcopy(self.optim.state_dict()),
                    'loss': loss_valid,
                }, self.validation_model_path)
            
            lim_valid = loss_valid

            # every 100 epochs, reload the model with lowest validation loss to date
            if epoch % 100 == 0:
            
                checkpoint = torch.load(self.validation_model_path, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
                lim_valid = checkpoint['loss'] # only keep new models that beat the best so far

                self.model.train()
