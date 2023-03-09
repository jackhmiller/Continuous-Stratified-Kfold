import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


class ContinuousSKFold:
    """
    Stratified K Fold for continuous data split
    ---
    
    folder = ContinuousSKFold(X=X,y=y)
    folds = folder.create_folds()
    
    """

    def __init__(self, X, y, n_folds=2):
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.ideal_bin_num = self.get_bin_num()

    def get_bin_num(self):
        start = min(self.y)
        stop = max(self.y)
        cut_dict = {n: np.digitize(self.y, bins=np.linspace(start, stop, num=n + 1)) for n in range(0, 20)}
        Y = pd.Series(self.y).rename('Y')
        avg = {nbins: Y.groupby(cut).mean().mean() for nbins, cut in cut_dict.items()}
        avg = pd.Series(avg.values(), index=avg.keys()).rename('mean_ybins').to_frame()
        final_bin_num = avg.mean()[0]

        try:
            assert final_bin_num < len(np.unique(self.y))
        except AssertionError:
            print('Number of bins greater than or equal to number of unique values')

        logger.info(f"Bin number for continuous stratified split: {round(final_bin_num)}")
        return round(final_bin_num)

    def merge_one_member_bins(self):
        bin_index = np.digitize(self.y, np.linspace(min(self.y), max(self.y), self.ideal_bin_num))
        y_binned_count = dict(Counter(bin_index))
        keys_with_single_value = []
        for key, value in y_binned_count.items():
            if value == 1:
                keys_with_single_value.append(key)

        for val in keys_with_single_value:
            nearest = self.find_nearest_bin([x for x in bin_index if x not in keys_with_single_value], val)
            ix_to_change = np.where(bin_index == val)[0][0]
            bin_index[ix_to_change] = nearest

        return bin_index

    @staticmethod
    def find_nearest_bin(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def create_folds(self) -> list:
        bin_index = self.merge_one_member_bins()
        train, test, _, _ = train_test_split(self.X,
                                             bin_index,
                                             stratify=bin_index,
                                             test_size=1/self.n_folds)
        folds = [{'train': train.index,
                  'test': test.index},
                 {'train': test.index,
                  'test': train.index}
                 ]

        return folds
