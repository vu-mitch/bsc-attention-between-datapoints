
import numpy as np
import torch
from tqdm import tqdm

from model.loss import Loss
from model.optim import LRScheduler
from model.optim import TradeoffAnnealer
from model.utils.logging_utils import Logger
from model.utils.encode_utils import torch_cast_to_dtype
from model.utils.batch_utils import collate_with_pre_batching
from model.preprocess import ColumnEncodingDataset, NPTDataset
from model.utils.eval_checkpoint_utils import EarlyStopCounter, EarlyStopSignal


class Trainer:
    def __init__(
            self, model, optimizer, scaler, args, wandb_run, cv_index,
            dataset: ColumnEncodingDataset = None,
            torch_dataset: NPTDataset = None,
            distributed_args=None):

        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = LRScheduler(
            c=args, name=args.exp_scheduler, optimizer=optimizer)
        self.c = args
        self.wandb_run = wandb_run
        self.is_distributed = False
        self.dataset = dataset
        self.torch_dataset = torch_dataset
        self.max_epochs = self.get_max_epochs()

        # Only needs to be set in distributed setting; otherwise, submodules
        # such as Loss and EarlyStopCounter use c.exp_device for tensor ops.
        self.gpu = None

        self.early_stop_counter = EarlyStopCounter(
            c=args, data_cache_prefix=dataset.model_cache_path,
            metadata=dataset.metadata, wandb_run=wandb_run, cv_index=cv_index,
            n_splits=min(dataset.n_cv_splits, args.exp_n_runs),
            device=self.gpu)

        # Initialize from checkpoint, if available
        num_steps = 0

        # Initialize tradeoff annealer, fast forward to number of steps
        # recorded in checkpoint.
        if self.c.exp_tradeoff != -1:
            self.tradeoff_annealer = TradeoffAnnealer(
                c=args, num_steps=num_steps)
        else:
            self.tradeoff_annealer = None

        self.logger = Logger(
            self.c, self.optimizer, self.gpu, self.tradeoff_annealer)

        self.loss = Loss(
            self.c, dataset.metadata,
            device=self.gpu, tradeoff_annealer=self.tradeoff_annealer,
            is_minibatch_sgd=self.c.exp_minibatch_sgd)

        if self.c.exp_eval_every_epoch_or_steps == 'steps':
            self.last_eval = 0

    def get_max_epochs(self):
        num_steps_per_epoch = self.get_num_steps_per_epoch()
        return int(
            np.ceil(self.c.exp_num_total_steps / num_steps_per_epoch))

    def get_num_steps_per_epoch(self):
        if self.c.exp_batch_size == -1:
            return 1

        N = self.dataset.metadata['N']
        return int(np.ceil(N / self.c.exp_batch_size))

    def train_and_eval(self):
        """Main training and evaluation loop."""
        self.logger.start_counting()

        curr_epoch = 1

        for epoch in range(1, self.max_epochs + 1):
            if self.per_epoch_train_eval(epoch=epoch):
                break

    def per_epoch_train_eval(self, epoch):
        early_stop = False

        # need to increase step counter by one here (because step counter is)
        # still at last step
        end_experiment = (
                self.scheduler.num_steps + 1 >= self.c.exp_num_total_steps)

        eval_model = end_experiment or self.eval_check(epoch)

        train_loss = self.run_epoch(dataset_mode='train', epoch=epoch,
                                    eval_model=False)

        if eval_model:
            early_stop = self.eval_model(train_loss, epoch, end_experiment)
        if early_stop or end_experiment:
            early_stop = True
            return early_stop

        return early_stop

    def eval_check(self, epoch):
        """Check if it's time to evaluate val and test errors."""

        if self.c.exp_eval_every_epoch_or_steps == 'epochs':
            return epoch % self.c.exp_eval_every_n == 0
        elif self.c.exp_eval_every_epoch_or_steps == 'steps':
            # Cannot guarantee that we hit modulus directly.
            if (self.scheduler.num_steps - self.last_eval >=
                    self.c.exp_eval_every_n):
                self.last_eval = self.scheduler.num_steps
                return True
            else:
                return False
        else:
            raise ValueError

    def run_epoch(self, dataset_mode, epoch, eval_model=False):
        print_n = self.c.exp_print_every_nth_forward

        if (dataset_mode == 'train') and not eval_model:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        # self.dataset.set_mode(mode=dataset_mode, epoch=epoch)

        # self.cv_dataset = next(self.dataset_gen) column_encoding_dataset.py
        tensor_dataset = NPTDataset(self.dataset.model_input, dataset_mode, batch_size=self.c.exp_batch_size)
        extra_args = {}

        if not self.c.data_set_on_cuda:
            extra_args['pin_memory'] = True

        batch_iter = torch.utils.data.DataLoader(
            dataset=tensor_dataset,
            shuffle=False,  # Already shuffled
            num_workers=0, # self.data_loader_nprocs
            collate_fn=collate_with_pre_batching,
            **extra_args)

        batch_iter = tqdm(
            batch_iter, desc='Batch') if self.c.verbose else batch_iter

        ##################################################################################
        # DATALOADER initialized, loop over batch
        ##################################################################################

        for batch_index, batch_dict_ in enumerate(batch_iter):
         # First step to work on batch
            # Normal execution
            if batch_dict_["input_arrs"][0].shape[0] == 0:
                break
            # print("batch_index", batch_index)
            # print("batch_dict_[\"input_arrs\"][0].shape[0]", batch_dict_["input_arrs"][0].shape[0])
            self.run_batch(
                batch_dict_, dataset_mode, eval_model,
                epoch, print_n, batch_index)

        if eval_model:

            loss_dict = self.loss.finalize_epoch_losses(eval_model)

        # Reset batch and epoch losses
        self.loss.reset()

        if (not eval_model) and self.c.exp_minibatch_sgd:
            loss_dict = None

        return loss_dict

    def run_batch(self, batch_dict, dataset_mode, eval_model,
                  epoch, print_n, batch_index):

        masked_tensors, label_mask_matrix, augmentation_mask_matrix = (
            batch_dict["masked_arrs"],
            batch_dict["label_mask_matrix"],
            batch_dict["augmentation_mask_matrix"])

        # Construct ground truth tensors
        ground_truth_tensors = batch_dict["input_arrs"]

        if not self.c.data_set_on_cuda:
            if self.is_distributed:
                device = self.gpu
            else:
                device = self.c.exp_device

            # Cast tensors to appropriate data type
            ground_truth_tensors = [
                torch_cast_to_dtype(obj=data, dtype_name=self.c.data_dtype)
                for data in ground_truth_tensors]
            ground_truth_tensors = [
                data.to(device=device, non_blocking=True)
                for data in ground_truth_tensors]
            masked_tensors = [
                data.to(device=device, non_blocking=True)
                for data in masked_tensors]

            # Send everything else used in loss compute to the device
            batch_dict["target_loss_matrix"] = (
                batch_dict["target_loss_matrix"].to(
                    device=device, non_blocking=True))

            if augmentation_mask_matrix is not None:
                augmentation_mask_matrix = augmentation_mask_matrix.to(
                    device=device, non_blocking=True)

            # Need label_mask_matrix for stochastic label masking
            if label_mask_matrix is not None:
                label_mask_matrix = label_mask_matrix.to(
                    device=device, non_blocking=True)

        forward_kwargs = dict(
            batch_dict=batch_dict,
            ground_truth_tensors=ground_truth_tensors,
            masked_tensors=masked_tensors, dataset_mode=dataset_mode,
            eval_model=eval_model, epoch=epoch,
            label_mask_matrix=label_mask_matrix,
            augmentation_mask_matrix=augmentation_mask_matrix)
        # for arr in ground_truth_tensors:
        #     print(arr.dtype)
        #     print(type(arr))
        # exit(0)
        # This Automatic Mixed Precision autocast is a no-op
        # of c.model_amp = False

        # with torch.cuda.amp.autocast(enabled=self.c.model_amp):
        self.forward_and_loss(**forward_kwargs)

        if (dataset_mode == 'train' and self.c.exp_minibatch_sgd
                and (not eval_model)):
            # Standardize and backprop on minibatch loss
            # if minibatch_sgd enabled
            loss_dict = self.loss.finalize_batch_losses()
            # print("loss_dict", loss_dict)
            train_loss = loss_dict['total_loss']
            # print("train_loss", train_loss)
            # exit(0)
            self.scaler.scale(train_loss).backward()
            self.scaler.step(self.optimizer)
            old_scaler = self.scaler.get_scale()
            # Updates the scale for next iteration.
            self.scaler.update()
            new_scaler = self.scaler.get_scale()

            if self.tradeoff_annealer is not None:
                self.tradeoff_annealer.step()

            if new_scaler < old_scaler:
                self.scheduler.step()

            self.optimizer.zero_grad()

            if print_n and (self.scheduler.num_steps % print_n == 0):
                self.logger.intermediate_log(
                    loss_dict=loss_dict,
                    num_steps=self.scheduler.num_steps,
                    batch_index=batch_index, epoch=epoch)

        # Update the epoch loss info with detached minibatch losses
        self.loss.update_losses(eval_model=eval_model)

    def forward_and_loss(
            self, batch_dict, ground_truth_tensors, masked_tensors,
            dataset_mode, eval_model, epoch, label_mask_matrix,
            augmentation_mask_matrix,):
        """Run forward pass and evaluate model loss."""
        extra_args = {}

        if eval_model:
            with torch.no_grad():
                output = self.model(masked_tensors, **extra_args)
        else: # true forward pass
            output = self.model(masked_tensors, **extra_args)

        loss_kwargs = dict(
            output=output, ground_truth_data=ground_truth_tensors,
            label_mask_matrix=label_mask_matrix,
            augmentation_mask_matrix=augmentation_mask_matrix,
            data_dict=batch_dict, dataset_mode=dataset_mode,
            eval_model=eval_model)

        self.loss.compute(**loss_kwargs)


    def eval_model(self, train_loss, epoch, end_experiment):
        """Obtain val and test losses."""
        kwargs = dict(epoch=epoch, eval_model=True)

        # Evaluate over val rows
        val_loss = self.run_epoch(dataset_mode='val', **kwargs)
        # print(f"val_loss: {val_loss} at epoch: {epoch}")
        if not (self.c.debug_eval_row_interactions and epoch == 2):
            # Early stopping check -- TODO: consider loss other than label?
            counter, best_model_and_opt = self.early_stop_counter.update(
                val_loss=val_loss['label']['total_loss'],
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=epoch,
                end_experiment=end_experiment,
                tradeoff_annealer=self.tradeoff_annealer)
        else:
            counter = EarlyStopSignal.END

        if not self.c.debug_eval_row_interactions:
            if (counter == EarlyStopSignal.STOP) or end_experiment:
                if best_model_and_opt is not None:
                    print('Loaded best performing model for last evaluation.')
                    self.model, self.optimizer, self.scaler, num_steps = (
                        best_model_and_opt)

                    # Initialize tradeoff annealer, fast forward to number of steps
                    # recorded in checkpoint.
                    if self.tradeoff_annealer is not None:
                        self.tradeoff_annealer = TradeoffAnnealer(
                            c=self.c, num_steps=num_steps)

                        # Update the tradeoff annealer reference in the logger
                        self.logger.tradeoff_annealer = self.tradeoff_annealer

                # update val loss
                val_loss = self.run_epoch(dataset_mode='val', **kwargs)

        if train_loss is None and not self.c.debug_eval_row_interactions:
            # Train and compute loss over masked features in train rows
            train_loss = self.run_epoch(dataset_mode='train', **kwargs)
        elif self.c.debug_eval_row_interactions:
            train_loss = {}

        # Check if we need to eval test
        if ((counter == EarlyStopSignal.STOP)
            or (not self.c.exp_eval_test_at_end_only)
                or (self.c.exp_eval_test_at_end_only and end_experiment)):
            # Evaluate over test and val rows again
            test_loss = self.run_epoch(dataset_mode='test', **kwargs)
        else:
            test_loss = None
        # print(f"test loss:{test_loss} at epooch:{epoch}")
        loss_dict = self.logger.log(
            train_loss, val_loss, test_loss, self.scheduler.num_steps, epoch)

        # Update summary metrics
        new_min = (
            self.early_stop_counter.num_inc_valid_loss_epochs == 0)
        self.logger.summary_log(loss_dict, new_min)

        if counter == EarlyStopSignal.STOP:
            print(self.early_stop_counter.stop_signal_message)
            return True
        else:
            return False

