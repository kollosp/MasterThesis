
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init

import numpy as np


from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import mean_absolute_error

import inspect
from numbers import Number
import math


class PytorchRegressor(BaseEstimator, RegressorMixin):
    """A pytorch regressor"""

    def __init__(self, output_dim=1, input_dim=100, hidden_layer_dims=[100, 100],
                 num_epochs=10, learning_rate=0.01, batch_size=32, shuffle=False,
                 callbacks=[], use_gpu=False, verbose=1):
        """
        Called when initializing the regressor
        """
        self._history = None
        self._model = None
        self._gpu = use_gpu and torch.cuda.is_available()

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def _build_model(self):
        self._history = None
        self._layer_dims = [self.input_dim] + \
            self.hidden_layer_dims + [self.output_dim]

        self._model = torch.nn.Sequential()

        # Loop through the layer dimensions and create an input layer, then
        # create each hidden layer with relu activation.
        for idx, dim in enumerate(self._layer_dims):
            if (idx < len(self._layer_dims) - 1):
                module = torch.nn.Linear(dim, self._layer_dims[idx + 1])
                init.xavier_uniform(module.weight)
                self._model.add_module("linear" + str(idx), module)

            if (idx < len(self._layer_dims) - 2):
                self._model.add_module("relu" + str(idx), torch.nn.ReLU())

        if self._gpu:
            self._model = self._model.cuda()

    def _train_model(self, X, y):

        torch_x = torch.from_numpy(X).float()
        torch_y = torch.from_numpy(y).float()
        if self._gpu:
            torch_x = torch_x.cuda()
            torch_y = torch_y.cuda()

        train = data_utils.TensorDataset(torch_x, torch_y)
        train_loader = data_utils.DataLoader(train, batch_size=self.batch_size,
                                             shuffle=self.shuffle)

        loss_fn = torch.nn.MSELoss(size_average=False)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate)

        self._history = {"loss": [], "val_loss": [], "mse_loss": []}

        finish = False
        for epoch in range(self.num_epochs):
            if finish:
                break


            loss = None
            idx = 0
            for idx, (minibatch, target) in enumerate(train_loader):
                y_pred = self._model(Variable(minibatch))

                loss = loss_fn(y_pred, Variable(
                    target.cuda().float() if self._gpu else target.float()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            y_labels = target.cpu().numpy() if self._gpu else target.numpy()
            y_pred_results = y_pred.cpu().data.numpy() if self._gpu else y_pred.data.numpy()
            error = mean_absolute_error(y_labels, y_pred_results)



            self._history["mse_loss"].append(loss.data)
            self._history["loss"].append(error)

            if self.verbose > 0:
                print("Results for epoch {}, loss {}, mse_loss {}".format(epoch + 1,
                                                                          error, loss.data))
            for callback in self.callbacks:
                callback.call(self._model, self._history)
                if callback.finish:
                    finish = True
                    break

    def fit(self, X, y, sample_weight=None):
        """
        Trains the pytorch regressor.
        """

        assert (type(self.input_dim) ==
                int), "input_dim parameter must be defined"
        assert (type(self.output_dim) == int), "output_dim must be defined"

        self._build_model()
        self._train_model(X, y)

        return self

    def predict(self, X, y=None):
        """
        Makes a prediction using the trained pytorch model
        """
        if self._history == None:
            raise RuntimeError("Regressor has not been fit")

        results = []
        split_size = math.ceil(len(X) / self.batch_size)

        # In case the requested size of prediction is too large for memory (especially gpu)
        # split into batchs, roughly similar to the original training batch size. Not
        # particularly scientific but should always be small enough.
        for batch in np.array_split(X, split_size):
            x_pred = Variable(torch.from_numpy(batch).float())
            y_pred = self._model(x_pred.cuda() if self._gpu else x_pred)
            y_pred_formatted = y_pred.cpu().data.numpy() if self._gpu else y_pred.data.numpy()
            results = np.append(results, y_pred_formatted)

        return results

    def score(self, X, y, sample_weight=None):
        """
        Scores the data using the trained pytorch model. Under current implementation
        returns negative mae.
        """
        y_pred = self.predict(X, y)
        return mean_absolute_error(y, y_pred) * -1

    def get_history(self):
        return self._history