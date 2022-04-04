import numpy as np

class XExtension:
    def __init__(self, method='direct'):
        method = 'derivative'
        self.method = method
        pass

    '''
        Function is used to adding arguments to X set. The basic arguments are coordinates, sometimes it is useful
        to add last y values (from previous moments). That is what this function is used for. 
        
        x_set is a set of normal X values
        additional is a set of all previous y values
        x_train is a set of unmodified x (if normalization is used)
        normalization is an nomalization class
        cv_indexes is an array of indexes selected by cross validation
    '''

    def apply(self, x_set, additional, x_train=None, normalization=None, cv_indexes = None):
        if self.method == 'none':
            return x_set

        #add additional to the set of features
        y_his_local = additional[cv_indexes]

        if self.method == 'direct':
            y_his_local = np.append(y_his_local, np.diff(y_his_local), axis=1)

        if self.method == 'derivative':
            #remove raw additional
            y_his_local =  np.diff(y_his_local)
            #y_his_local = np.append(y_his_local, np.diff(y_his_local), axis=1)

        if self.method == 'mean':
            #print(y_his_local)
            #print(np.mean(y_his_local, axis=0)[0:1])

            k = np.tile([np.mean(y_his_local, axis=0)], (y_his_local.shape[0],1))
            y_his_local = np.append(k, np.diff(y_his_local), axis=1)

            #y_his_local = np.append(y_his_local, np.tile([np.std(y_his_local)], (y_his_local.shape[0],1)), axis=1)
            #y_his_local = np.append(y_his_local, k, axis=1)
            #y_his_local = np.append(y_his_local, diff, axis=1)

        #create a vector after normalization

        if normalization is not None and x_train is not None:
            additional = np.zeros((normalization.get_grid_size(), y_his_local.shape[1]))
            for k in range(y_his_local.shape[1]):
                _, additional[:, k] = normalization.apply(x_train, y_his_local[:, k])
            y_his_local = additional

        return np.append(x_set.copy(), y_his_local, axis=1)

