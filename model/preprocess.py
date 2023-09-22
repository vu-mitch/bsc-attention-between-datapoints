import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque

from abc import ABC
import inspect
from abc import ABCMeta

from torch.utils.data import ConcatDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

class BaseDataset(ABC):
    def __init__(
            self, fixed_test_set_index):
        """
        Args:
            fixed_test_set_index: int, if specified, the dataset has a
                fixed test set starting at this index. Needed for
                comparability to other methods.
        """
        self.fixed_test_set_index = fixed_test_set_index
        self.c = None
        self.data_table = None
        self.missing_matrix = None
        self.N = None
        self.D = None
        self.cat_features = None
        self.num_features = None
        self.cat_target_cols = None
        self.num_target_cols = None
        self.auroc_setting = None
        self.fixed_split_indices = None

    @staticmethod
    def load_data(dataset_name):
        path = Path(__file__).parent.parent / "datasets" / dataset_name
        df = pd.read_csv(f"{path}.csv")
        return df

    def use_auroc(self, force_disable=None):
        """
        Disable AUROC metric:
            (i)  if we do not have a single categorical target column,
            (ii) if the single categorical target column is multiclass.
        """

        disable = 'Disabling AUROC metric.'

        if force_disable:
            print(disable)
            return False

        if not self.c.metrics_auroc:
            print(disable)
            print("As per config argument 'metrics_auroc'.")
            return False

        num_target_cols, cat_target_cols = (
            self.num_target_cols, self.cat_target_cols)
        n_cat_target_cols = len(cat_target_cols)

        if n_cat_target_cols != 1:
            print(disable)
            print(
                f'\tBecause dataset has {n_cat_target_cols} =/= 1 '
                f'categorical target columns.')
            if n_cat_target_cols > 1:
                print(
                    '\tNote that we have not decided how we want to handle '
                    'AUROC among multiple categorical target columns.')
            return False
        elif num_target_cols:
            print(disable)
            print(
                '\tBecause dataset has a nonzero count of '
                'numerical target columns.')
            return False
        else:
            auroc_col = cat_target_cols[0]
            if n_classes := len(np.unique(self.data_table[:, auroc_col])) > 2:
                print(disable)
                print(f'\tBecause AUROC does not (in the current implem.) '
                      f'support multiclass ({n_classes}) classification.')
                return False

        return True

    @staticmethod
    def filter_column_types(df):
        object_columns = list((df.select_dtypes(include=['object'])).columns)
        cat_features = [df.columns.get_loc(col) for col in object_columns if col in df]
        num_features = list(set(np.arange(len(df.columns))).difference(cat_features))
        return cat_features, num_features


class ColumnEncodingDataset:
    def __init__(self, args, dataset):
        super(ColumnEncodingDataset).__init__()
        self.args = args
        self.dataset = dataset
        self.encoded_input = list()
        self.masks, self.label_bert_masks = None, dict()
        self.model_input = None
        self.dataset_gen = None
        self.metadata = None
        self.input_feature_dims = list()
        self.train_mask_matrix = None
        self.val_mask_matrix = None
        self.test_mask_matrix = None
        self.standardisation, self.sigmas = None, None
        self.n_cv_splits = int(1 / args.exp_test_perc)
        self.cache_path, self.model_cache_path = self.init_cache_path()

        self.reset_cv_splits()
        self.create_metadata()

    def create_metadata(self):
        self.metadata = \
            {"N": self.dataset.N, "D": self.dataset.D, "cat_features": self.dataset.cat_features,
             "num_features": self.dataset.num_features, "cat_target_cols": self.dataset.cat_target_cols,
             "num_target_cols": self.dataset.num_target_cols, "input_feature_dims": self.input_feature_dims,
             "fixed_test_set_index": None, "auroc_setting": True}

    def init_cache_path(self):
        ssl_str = f'ssl__{self.args.model_is_semi_supervised}'

        cache_path = os.path.join(
            self.args.data_path, self.args.data_set, ssl_str,
            f'np_seed={self.args.np_seed}__n_cv_splits={self.n_cv_splits}'
            f'__exp_num_runs={self.args.exp_n_runs}')

        if self.args.model_checkpoint_key is not None: # model_checkpoint_key=None
            model_cache_path = os.path.join(
                cache_path, self.args.model_checkpoint_key)
        else:
            model_cache_path = cache_path

        if not os.path.exists(cache_path):
            try:
                os.makedirs(cache_path)
            except FileExistsError as e:
                print(e)

        if not os.path.exists(model_cache_path):
            try:
                os.makedirs(model_cache_path)
            except FileExistsError as e:
                print(e)

        return cache_path, model_cache_path

    def reset_cv_splits(self):
        self.dataset_gen = self.run_preprocessing()
        self.curr_cv_split = -1

    def load_next_cv_split(self):
        self.curr_cv_split += 1
        next(self.dataset_gen)

    def run_preprocessing(self):
        self.dataset_gen = self.generate_dataset()
        for data_dict in self.dataset_gen:

            row_index_order = self.create_masks(data_dict["range_indices"])
            num_train_indices = len(data_dict["range_indices"][0])

            self.encode(data_dict["data_table"], num_train_indices)
            # print("self.encoded_input\n", torch.hstack(self.encoded_input[1:]))
            self.model_input = {dataset_mode: self.mask_according_to_mode(dataset_mode, row_index_order)
                                for dataset_mode in ["train", "val", "test"]}

            yield self.model_input

    def generate_dataset(self):
        train_val_test_splits = self.get_splits()
        for train_indices, val_indices, test_indices in train_val_test_splits:
            new_data_table = np.concatenate([
                self.dataset.data_table[train_indices],
                self.dataset.data_table[val_indices],
                self.dataset.data_table[test_indices]], axis=0)

            lens = np.cumsum([0] + [len(i) for i in [train_indices, val_indices, test_indices]])
            range_indices = [list(range(lens[i], lens[i + 1])) for i in range(len(lens) - 1)]
            self.row_boundaries = {'train': lens[1], 'val': lens[2], 'test': lens[3]}

            data_dict = dict(data_table=new_data_table, range_indices=range_indices)
            yield data_dict

    def get_splits(self):
        should_stratify = (len(self.dataset.cat_target_cols) == 1
                           and len(self.dataset.num_target_cols) == 0)
        label_col = self.dataset.data_table[:, self.dataset.cat_target_cols] if should_stratify else np.arange(self.dataset.N)
        kf_class = StratifiedKFold if should_stratify else KFold
        kf = kf_class(n_splits=self.n_cv_splits, shuffle=True, random_state=self.args.np_seed)

        train_test_splits = kf.split(range(self.dataset.N), label_col)

        # START OF GENERATOR
        for train_val_indices, test_indices in train_test_splits:
            train_val_label_rows = label_col[train_val_indices] if should_stratify else None

            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=self.args.exp_val_perc, shuffle=True,
                random_state=21, stratify=train_val_label_rows)

            yield train_indices, val_indices, test_indices

    def create_masks(self, range_indices):
        self.train_mask_matrix, self.val_mask_matrix, self.test_mask_matrix = [
            self.get_matrix_from_rows(
                rows=mode_indices,
                cols=self.dataset.cat_target_cols + self.dataset.num_target_cols,
                N=self.dataset.N, D=self.dataset.D)
            for mode_indices in range_indices]

        missing_matrix = torch.tensor(self.dataset.missing_matrix)

        self.masks = {key: matrix for key, matrix in zip(
            ("train", "val", "test"), (self.train_mask_matrix, self.val_mask_matrix, self.test_mask_matrix))}

        for mode in ["train", "val", "test"]:
            if self.args.model_is_semi_supervised:
                bert_mask_matrix = ~(self.train_mask_matrix | self.val_mask_matrix |
                                     self.test_mask_matrix | missing_matrix)
                label_mask_matrix = torch.zeros_like(bert_mask_matrix) == 1
                label_mask_matrix[:, self.dataset.cat_target_cols + self.dataset.num_target_cols] = True
                self.label_bert_masks[mode] = label_mask_matrix, bert_mask_matrix
            else:
                if mode == "train":
                    label_mask_matrix = self.masks[mode]
                else:
                    label_mask_matrix = label_mask_matrix | self.masks[mode]
                bert_mask_matrix = ~(label_mask_matrix | missing_matrix)
                self.label_bert_masks[mode] = (label_mask_matrix, bert_mask_matrix)

        row_index_order = np.concatenate(range_indices)
        # np.random.shuffle(row_index_order)

        return row_index_order

    @staticmethod
    def get_matrix_from_rows(rows, cols, N, D):
        matrix = torch.zeros(size=(N, D), dtype=torch.bool)
        for col in cols:
            matrix[rows, col] = True

        return matrix

    def encode(self, data_table, num_train_indices):
        # normalize numeric entries
        val_mask, test_mask = self.val_mask_matrix.numpy(), self.test_mask_matrix.numpy()
        # only use fit on values that are not missing and exclude labels from val and test
        encoding_matrix_to_use_fit_function = (1 - self.dataset.missing_matrix -
                                               val_mask - test_mask).astype(np.bool_)

        if not self.args.model_is_semi_supervised:
            encoding_matrix_to_use_fit_function[num_train_indices:] = False

        non_missing_matrix = ~self.dataset.missing_matrix

        self.standardisation = np.nan * np.ones((self.dataset.D, 2))
        self.sigmas = []

        for col_idx in range(self.dataset.D):
            column = data_table[:, col_idx].reshape(-1, 1)
            enc_fit_filter = encoding_matrix_to_use_fit_function[:, col_idx] # deal with missing entry values
            enc_fit_col = self.dataset.data_table[enc_fit_filter, col_idx].reshape(-1, 1)

            non_missing_filter = non_missing_matrix[:, col_idx]
            non_missing_col = data_table[non_missing_filter, col_idx].reshape(-1, 1)

            if col_idx in self.dataset.cat_features:
                fitted_enc = OneHotEncoder(sparse=False).fit(non_missing_col)
                encoded_col = fitted_enc.transform(non_missing_col)
                self.sigmas.append(-1)
            else:
                fitted_enc = StandardScaler().fit(enc_fit_col)
                encoded_col = (fitted_enc.transform(non_missing_col)).reshape(-1, 1)
                self.standardisation[col_idx, 0] = fitted_enc.mean_[0]
                self.standardisation[col_idx, 0] = fitted_enc.scale_[0]
                self.sigmas.append(fitted_enc.scale_[0])


            encoded_col = self.insert_missing_values(non_missing_filter, encoded_col)

            if self.args.model_bert_augmentation:
                encoded_col = np.hstack([encoded_col, np.zeros((self.dataset.N, 1))])
                missing_filter = self.dataset.missing_matrix[:, col_idx]
                # Zero out all one-hots or single numerical values
                encoded_col[missing_filter, :] = 0
                # Set mask token to 1 into expanded dimenstion
                encoded_col[missing_filter, -1] = 1

            encoded_col = torch.tensor(encoded_col, device=torch.device("cpu"))

            if col_idx in self.dataset.cat_features:
                encoded_col = encoded_col == 1 # get boolean matrix
            else:
                encoded_col.to(torch.float64)

            self.encoded_input.append(encoded_col)
            self.input_feature_dims.append(encoded_col.shape[1])

    @staticmethod
    def insert_missing_values(
            non_missing_col_filter, encoded_non_missing_col_values):
        encoded_col = []
        encoded_queue = deque(encoded_non_missing_col_values)
        one_hot_length = encoded_non_missing_col_values.shape[1]

        for elem_was_encoded in non_missing_col_filter:
            if elem_was_encoded:
                encoded_col.append(encoded_queue.popleft())
            else:
                encoded_col.append(np.zeros(one_hot_length))

        return np.array(encoded_col)
    def mask_according_to_mode(self, dataset_mode, row_index_order):
        stochastic_label_masks = None
        # choose target matrix according to dataset_mode
        target_loss_matrix = self.masks[dataset_mode][row_index_order, :]

        if self.args.model_label_bert_mask_prob[dataset_mode] < 1:
            stochastic_label_masks = {key: self.masks[key][row_index_order, :] for key in ["train", "val", "test"]}

        to_mask_input = [col[row_index_order, :] for col in self.encoded_input]
        to_mask_mode, to_mask_bert = self.label_bert_masks[dataset_mode]
        to_mask_mode, to_mask_bert = to_mask_mode[row_index_order, :], to_mask_bert[row_index_order, :]

        masked_tensors, label_mask_matrix, augmentation_mask_matrix = (
            self.mask_data_for_dataset_mode(
                to_mask_mode,
                stochastic_label_masks,
                self.args, self.dataset.cat_features, to_mask_bert,
                to_mask_input, dataset_mode, torch.device("cpu")))

        if not self.args.model_is_semi_supervised:
            threshold = self.row_boundaries[dataset_mode]

            to_mask_input = [col[:threshold] for col in to_mask_input]
            masked_tensors = [col[:threshold] for col in masked_tensors]
            target_loss_matrix = target_loss_matrix[:threshold, :]

            if label_mask_matrix is not None:
                label_mask_matrix = label_mask_matrix[:threshold, :]

            if augmentation_mask_matrix is not None:
                augmentation_mask_matrix = augmentation_mask_matrix[:threshold, :]

        return to_mask_input, masked_tensors, label_mask_matrix, \
               augmentation_mask_matrix, target_loss_matrix, self.sigmas

    def mask_data_for_dataset_mode(self,
                                deterministic_label_masks,
                                stochastic_label_masks,
                                args, cat_features, bert_mask_matrix,
                                data_arrs, dataset_mode, device):

        input_arrs = [arr.clone() for arr in data_arrs]
        # ******* 1 TARGET MASKING ********
        if args.model_label_bert_mask_prob[dataset_mode] == 1:
            mask_candidates = deterministic_label_masks

            masked_arrs, label_mask_matrix = (
                self.apply_mask(
                    data_arrs=input_arrs,
                    mask_candidates=mask_candidates,
                    cat_features=cat_features,
                    mask_prob=1,
                    args=args))
            label_mask_matrix = None # Set this to none because label_mask_matrix is all possible values.

        else:
            # Stochastic Masking on Some Labels
            label_prob = args.model_label_bert_mask_prob[dataset_mode] # 0.15

            if (label_prob == 0) and (dataset_mode == 'train'):
                # mask none of the train labels but need to make sure we tell that to the loss also
                # (no longer want to compute loss here) --> empty mask_indices matrix
                stochastic_mask_indices = np.zeros_like(stochastic_label_masks['train'])
                masked_arrs = input_arrs

            elif label_prob > 0:
                # Stochastically mask out some targets (never test targets)
                DATA_MODE_TO_LABEL_BERT_MODE = {'train': ['train'], 'val': ['train'], 'test': ['train', 'val']}

                mask_categories = DATA_MODE_TO_LABEL_BERT_MODE[dataset_mode]
                mask_candidates = torch.stack([stochastic_label_masks[category] for
                                               category in mask_categories]).sum(0, dtype=torch.bool)

                masked_arrs, stochastic_mask_indices = self.apply_mask(
                    data_arrs=input_arrs,
                    mask_candidates=mask_candidates,
                    cat_features=cat_features,
                    mask_prob=label_prob,
                    args=args)

                # label_mask_indices not outside mask_candidates
                assert ((~mask_candidates) & (stochastic_mask_indices)).any().item() is False
                if args.model_label_bert_mask_prob[dataset_mode] == 1:
                    assert (mask_candidates & stochastic_mask_indices).all().item() is True
            else:
                masked_arrs = input_arrs

            # When we do stochastic label masking, some labels will be
            # masked out deterministically, to avoid information leaks.
            DATA_MODE_TO_LABEL_BERT_FIXED = {
                'train': ['val', 'test'],
                'val': ['val', 'test'],
                'test': ['test'],
            }
            # Deterministic masking on some labels (always test targets)
            mask_categories = DATA_MODE_TO_LABEL_BERT_FIXED[dataset_mode]
            mask_candidates = torch.stack([stochastic_label_masks[category]
                                           for category in mask_categories]).sum(0, dtype=torch.bool)

            label_mask_matrix = stochastic_mask_indices if dataset_mode == 'train' else None

        # ****** 2 – FEATURE MASKING *******
        if args.model_augmentation_bert_mask_prob[dataset_mode] > 0:
            masked_arrs, augmentation_mask_matrix = (
                self.apply_mask(
                    data_arrs=masked_arrs,
                    mask_candidates=bert_mask_matrix,
                    cat_features=cat_features,
                    mask_prob=(args.model_augmentation_bert_mask_prob[dataset_mode]),
                    args=args)) # at train: 0.15 else 0
        else:
            augmentation_mask_matrix = None

        masked_arrs = [arr.type(torch.float32) for arr in masked_arrs]

        return masked_arrs, label_mask_matrix, augmentation_mask_matrix

    @staticmethod
    def apply_mask(data_arrs, mask_candidates, cat_features, mask_prob, args):
        device = torch.device("cpu")
        bert_random_mask = None

        num_examples = data_arrs[0].shape[0]
        if not args.model_is_semi_supervised:
            mask_candidates = mask_candidates[:num_examples]

        if mask_prob == 1:
            mask = mask_candidates
        else:
            # Stochastic Bert Style Masking
            # extract a list of all matrix entries with mask candidates
            mask_entries = torch.nonzero(mask_candidates, as_tuple=False) # shape(num_rows, 2)
            d = len(mask_entries)
            # Performing Nm bernoulli samples with probability mask_prob with std deviation of
            std = np.sqrt(d * mask_prob * (1 - mask_prob))
            # number of masks sampled
            num_masks_sampled = int(mask_prob * d + np.random.normal(0, std))
            num_masks_sampled = max(min(num_masks_sampled, d), 0)
            # take a random subset of the total number of mask indices
            mask_indices = np.random.choice(
                    np.arange(0, d),
                    size=num_masks_sampled,
                    replace=False)
            # Select valid indices from sampled mask indices: shape(len(mask_indices, 2)
            sampled_mask = mask_entries[mask_indices, :]
            # split mask indices into entries to zero out and entries to randomly resample
            bert_random_mask_proportion = 1 - int(
                args.model_bert_mask_percentage * len(sampled_mask))
            # select indices after mask indices are chosen
            bert_random_mask_indices = sampled_mask[:bert_random_mask_proportion]
            # Reconstruct mask matrices from list of entries.
            mask = torch.sparse.FloatTensor(
                sampled_mask.T,
                torch.ones(len(sampled_mask), dtype=torch.int, device=device),
                mask_candidates.size(),
            ).to_dense().type(torch.bool)

            bert_random_mask = torch.sparse.FloatTensor(
                bert_random_mask_indices.T,
                torch.ones(
                    len(bert_random_mask_indices), dtype=torch.int, device=device),
                mask_candidates.size(),
            ).to_dense().type(torch.bool)
            # Mask is never 1 where mask_candidate is 0
            assert ((mask_candidates.int() - mask.int()) < 0).sum() == 0
            # bert_random_mask is never 1 where Mask is 0
            assert ((mask.int() - bert_random_mask.int()) < 0).sum() == 0

        # Iterate over all columns and set mask.
        for col, data_arr in enumerate(data_arrs):
            # Get boolean 'mask' selection mask for this column.
            mask_col = mask[:, col]
            # If no masks in this column, continue.
            if mask_col.sum() == 0:
                continue
            # zero out indices according to bert masking
            data_arr[mask_col, :] = 0
            # if None all mask entries are given a "1" mask token
            if bert_random_mask is None:
                data_arr[mask_col, -1] = 1
                continue

            bert_random_mask_col = bert_random_mask[:, col]
            # Determine for which entries to set the mask token.
            # If bert_random_mask_col is all False mask entries determined by mask_col
            bert_mask_token_col = mask_col & (~bert_random_mask_col)
            # Set mask token in last position of each feature vector,
            # for appropriate entries.
            data_arr[bert_mask_token_col, -1] = 1

            if bert_random_mask_col.sum() == 0:
                continue

            if col in cat_features: # for categorical features select random columns
                # for all selected entries in row and set to one
                random_cols = torch.randint(
                    low=0,
                    high=data_arr.shape[1] - 1,  # high is exclusive for torch
                    size=[bert_random_mask_col.sum()],
                    requires_grad=False)

                data_arr[bert_random_mask_col, random_cols] = 1
            else:# For continuous features sample new entry values from normal distribution.
                data_arr[bert_random_mask_col, 0] = torch.normal(
                    mean=0, std=1,
                    size=[bert_random_mask_col.sum()], dtype=torch.float64,
                    device=device)

        return data_arrs, mask


class ConcreteDataset(BaseDataset):# regression dataset
    def __init__(self, c):
        super(ConcreteDataset, self).__init__(
            fixed_test_set_index=None)
        self.c = c
        self.data_table = self.load_data(c.data_set).to_numpy()
        self.N, self.D = self.data_table.shape
        self.cat_features = []
        self.num_features = list(range(self.D))
        self.cat_target_cols = []
        self.num_target_cols = [self.D - 1]
        self.missing_matrix = np.zeros(shape=(self.N, self.D), dtype=np.bool_)

class BreastCancerDataset(BaseDataset): # binary classification
    def __init__(self, c):
        super(BreastCancerDataset, self).__init__(
            fixed_test_set_index=None)
        self.c = c
        self.data_table = self.load_data(c.data_set).to_numpy()[:, 1:-1]
        self.N, self.D = self.data_table.shape
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.cat_features = [0]
        self.num_features = list(range(1, self.D))
        self.cat_target_cols = [0]
        self.num_target_cols = []

class MinicancerDataset(BreastCancerDataset):
    """Class imbalance is [357, 212]."""
    def __init__(self, c):
        super().__init__(c)
        dm = self.data_table[self.data_table[:, 0] == 'M'][:8, :5]
        db = self.data_table[self.data_table[:, 0] == 'B'][:8, :5]
        self.data_table = np.concatenate([dm, db], 0)
        self.N, self.D = self.data_table.shape
        self.num_features = list(range(1, self.D))
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.missing_matrix[0, 0] = True # missing label
        self.missing_matrix[3, 1] = True # missing numeric feature entry
        self.missing_matrix[2, 2] = True

class IncomeDataset(BaseDataset): # binary classification
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c
        df = self.load_data(c.data_set) # prefer to work with pandas for filtering column types
        self.N, self.D = df.shape
        self.missing_matrix = df.eq("?").to_numpy()
        self.cat_features, self.num_features = self.filter_column_types(df)
        self.cat_target_cols = [self.D - 1]
        self.num_target_cols = []
        self.data_table = df.to_numpy()

class MiniadultDataset(IncomeDataset):
    def __init__(self, c):
        super().__init__(c)
        self.c = c
        df = self.load_data(c.data_set).to_numpy()
        dm = df[df[:, -1] == '>50K'][:8, :]
        db = df[df[:, -1] == '<=50K'][:8, :]
        self.data_table = np.concatenate([dm, db], 0)
        self.N, self.D = self.data_table.shape
        self.missing_matrix = pd.DataFrame(self.data_table).eq("?").to_numpy()
        self.cat_target_cols = [self.D - 1]
        self.num_target_cols = []

class CarkickDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c
        df = self.load_data(c.data_set)
        self.N, self.D = df.shape
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.cat_features, self.num_features = self.filter_column_types(df)
        self.cat_target_cols = [self.D - 1]
        self.num_target_cols = []
        self.data_table = df.to_numpy()

class NPTDataset(torch.utils.data.IterableDataset):
    def __init__(self, model_input, dataset_mode, batch_size):
        super(NPTDataset).__init__()
        self.encoded_input = model_input[dataset_mode][0]
        self.masked_tensors = model_input[dataset_mode][1]
        self.label_mask_matrix = model_input[dataset_mode][2]
        self.augmentation_mask_matrix = model_input[dataset_mode][3]
        self.target_loss_matrix = model_input[dataset_mode][4]
        self.sigmas = model_input[dataset_mode][5]
        self.batch_size = batch_size
        self.row_index, self.batch_index = 0, 0
        # self.test()

    def __len__(self):
        return int(np.ceil(self.masked_tensors[0].shape[0] / self.batch_size))

    def __iter__(self):
        return self
        # input_batch = [col[self.row_index:self.row_index + self.batch_size]
        #                for col in self.encoded_input]
        # masked_batch = [col[self.row_index:self.row_index + self.batch_size]
        #                 for col in self.masked_tensors]
        # target_loss_matrix = self.target_loss_matrix[self.row_index:self.row_index + self.batch_size]
        #
        # label_mask_matrix, augmentation_mask_matrix = None, None
        # if self.label_mask_matrix is not None:
        #     label_mask_matrix = self.label_mask_matrix[self.row_index:self.row_index + self.batch_size]
        # if self.augmentation_mask_matrix is not None:
        #     augmentation_mask_matrix = self.augmentation_mask_matrix[self.row_index:self.row_index + self.batch_size]
        #
        # self.row_index += self.batch_size
        # batch_dict = {"input_arrs": input_batch, "masked_arrs": masked_batch,
        #               "target_loss_matrix": target_loss_matrix, "label_mask_matrix": label_mask_matrix,
        #               "augmentation_mask_matrix": augmentation_mask_matrix, "sigmas": self.sigmas}
        #
        # yield self.encoded_input

    def __next__(self):
        input_batch = [col[self.row_index:self.row_index + self.batch_size]
                        for col in self.encoded_input]
        masked_batch = [col[self.row_index:self.row_index + self.batch_size]
                        for col in self.masked_tensors]
        target_loss_matrix = self.target_loss_matrix[self.row_index:self.row_index + self.batch_size]

        label_mask_matrix, augmentation_mask_matrix = None, None
        if self.label_mask_matrix is not None:
            label_mask_matrix = self.label_mask_matrix[self.row_index:self.row_index + self.batch_size]
        if self.augmentation_mask_matrix is not None:
            augmentation_mask_matrix = self.augmentation_mask_matrix[self.row_index:self.row_index + self.batch_size]

        self.row_index += self.batch_size
        self.batch_index += 1
        batch_dict = {"input_arrs": input_batch, "masked_arrs": masked_batch,
                      "target_loss_matrix": target_loss_matrix, "label_mask_matrix": label_mask_matrix,
                      "augmentation_mask_matrix": augmentation_mask_matrix, "sigmas": self.sigmas,
                      "row_idx": self.row_index, "batch_idx": self.batch_index}

        return batch_dict


def find_dataset(name: str) -> ABCMeta:
    """
    Get specified dataset using a substring matching procedure.

    :param name: substring of dataset name
    :return: specified dataset if found
    """
    possible_datasets = inspect.getmembers(
        object=sys.modules[__name__],
        predicate=lambda entity: inspect.isclass(entity) and name.lower() in entity.__name__.lower()
    )
    print("possible_datasets", possible_datasets)
    assert len(possible_datasets) > 0, r'No dataset with query <{name}> found!'
    assert len(possible_datasets) == 1, r'Ambiguous dataset query <{name}>!'
    return possible_datasets[0][1]