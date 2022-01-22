import numpy as np
from scipy import stats

class Postprocessing:
    @staticmethod
    def process(data):
        d = np.zeros([5, data.shape[1], data.shape[1]])
        alpha = 0.05

        # every dataset
        for i in range(len(data)):
            # every classifier
            ##NxN table
            for k1 in range(len(data[i])):
                for k2 in range(len(data[i])):
                    s = stats.ttest_ind(data[i, k1], data[i, k2])
                    # sprawdzeie jak bardzo rozklady dwoch zmiennych losowych sa od siebie odlegle
                    d[0, k1, k2] = s.statistic
                    # prawdopodobienstwo spelnienia proby zerowej
                    d[1, k1, k2] = s.pvalue
                    # wybor lepszych wynikow
                    d[2, k1, k2] = s.statistic > 0
                    # weryfikacja wynikow z progriem (biezemy pod uwage tylko wartosci lepsze niz...)
                    d[3, k1, k2] = s.pvalue < alpha
                    # ktory jest lepszy oraz jest znacząco lepszy staystycznie
                    d[4, k1, k2] = d[3, k1, k2] * d[2, k1, k2]
        return d

