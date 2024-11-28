import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner import BaseLoop
from mmengine.utils import is_list_of

def calc_dynamic_intervals(
    start_interval: int,
    dynamic_interval_list: Optional[List[Tuple[int, int]]] = None
) -> Tuple[List[int], List[int]]:
    """Calculate dynamic intervals.

    Args:
        start_interval (int): The interval used in the beginning.
        dynamic_interval_list (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.

    Returns:
        Tuple[List[int], List[int]]: a list of milestone and its corresponding
        intervals.
    """
    if dynamic_interval_list is None:
        return [0], [start_interval]

    assert is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals

class _InfiniteDataloaderIterator:
    """An infinite dataloader iterator wrapper for IterBasedTrainLoop.

    It resets the dataloader to continue iterating when the iterator has
    iterated over all the data. However, this approach is not efficient, as the
    workers need to be restarted every time the dataloader is reset. It is
    recommended to use `mmengine.dataset.InfiniteSampler` to enable the
    dataloader to iterate infinitely.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader
        self._iterator = iter(self._dataloader)
        self._epoch = 0

    def __iter__(self):
        return self

    def __next__(self) -> Sequence[dict]:
        try:
            data = next(self._iterator)
        except StopIteration:
            print_log(
                'Reach the end of the dataloader, it will be '
                'restarted and continue to iterate. It is '
                'recommended to use '
                '`mmengine.dataset.InfiniteSampler` to enable the '
                'dataloader to iterate infinitely.',
                logger='current',
                level=logging.WARNING)
            self._epoch += 1
            if hasattr(self._dataloader, 'sampler') and hasattr(
                    self._dataloader.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no sampler,
                # or data loader uses `SequentialSampler` in Pytorch.
                self._dataloader.sampler.set_epoch(self._epoch)

            elif hasattr(self._dataloader, 'batch_sampler') and hasattr(
                    self._dataloader.batch_sampler.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no batch
                # sampler. batch sampler in pytorch warps the sampler as its
                # attributes.
                self._dataloader.batch_sampler.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self._iterator = iter(self._dataloader)
            data = next(self._iterator)
        return data

class TrainLoop(BaseLoop):
    """
    Args:
        runner (Runner): A reference of the runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_iters (Optional[int]): Total training iterations.
        max_epochs (Optional[Union[int, float]]): Total training epochs.
        val_begin (int): The iteration that begins validation. Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
        dynamic_intervals (List[Tuple[int, int]], optional): The first element
            in the tuple is a milestone and the second element is an interval.
            The interval is used after the corresponding milestone. Defaults to None.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 max_iters: Optional[int] = None,
                 max_epochs: Optional[Union[int, float]] = None,
                 val_begin: int = 1,
                 val_interval: int = 1000,
                 dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        
        # print(" " * 6 + "Inside TrainLoop init")
        # print(" " * 7 + "↓")
        # print(" " * 7 + "↓")
        
        if max_iters is None and max_epochs is None:
            raise RuntimeError('Please specify either `max_iters` or `max_epochs` in `train_cfg`.')
        elif max_iters is not None and max_epochs is not None:
            raise RuntimeError('Only one of `max_iters` or `max_epochs` can be set in `train_cfg`.')
        
        if max_iters is not None:
            self._max_iters = int(max_iters)
            assert self._max_iters == max_iters, f'`max_iters` should be an integer, but got {max_iters}.'
        else:
            if isinstance(dataloader, dict):
                diff_rank_seed = runner._randomness_cfg.get('diff_rank_seed', False)
                dataloader = runner.build_dataloader(dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
            self._max_iters = int(max_epochs * len(dataloader))
        
        super().__init__(runner, dataloader)
        self._max_epochs = 1  # Compatibility with epoch-based training
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        self.stop_training = False
        
        # Dataset metainfo setup
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = self.dataloader.dataset.metainfo
        else:
            print_log(f'Dataset {self.dataloader.dataset.__class__.__name__} has no metainfo. '
                      'Setting `dataset_meta` in visualizer to None.',
                      logger='current', level=logging.WARNING)
        
        # Setup dataloader iterator and dynamic intervals
        self.dataloader_iterator = _InfiniteDataloaderIterator(self.dataloader)
        self.dynamic_milestones, self.dynamic_intervals = calc_dynamic_intervals(
            self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> None:
        """Launch training loop."""
        self.runner.call_hook('before_train')
        self.runner.call_hook('before_train_epoch')
        
        if self._iter > 0:
            print_log(f'Advancing dataloader {self._iter} steps to skip data already trained.',
                      logger='current', level=logging.WARNING)
            for _ in range(self._iter):
                next(self.dataloader_iterator)
        
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()
            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook('before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        
        #print("Data Batch:", data_batch)
        #print()
        outputs = self.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook('after_train_iter', batch_idx=self._iter, data_batch=data_batch, outputs=outputs)
        self._iter += 1

    def train_step(self, data, optim_wrapper):
        data = self.runner.model.data_preprocessor(data)
        loss = self.runner.model.forward(**data, mode='loss')
        #logits_dict = self.runner.model.forward(**data, mode='predict')
        #print("logits: ", logits_dict)
        parsed_losses, log_vars = self.runner.model.parse_losses(loss)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the `val_interval`."""
        step = bisect.bisect(self.dynamic_milestones, (self._iter + 1))
        self.val_interval = self.dynamic_intervals[step - 1]

