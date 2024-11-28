import bisect
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner import BaseLoop
from mmengine.utils import is_list_of

# for visualizing llm output during training
from transformers import GenerationConfig, StoppingCriteriaList
from mmengine.utils.misc import get_object_from_string
from xtuner.dataset.llast import prepare_inputs_labels_for_llast
from xtuner.registry import BUILDER
from xtuner.utils import StopWordStoppingCriteria
from mmengine.model import is_model_wrapper
from mmengine.runner import autocast

def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria

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
    """Unified class for iter-based and epoch-based training loops.

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
                 tokenizer,
                 dataloader: Union[DataLoader, Dict],
                 max_iters: Optional[int] = None,
                 max_epochs: Optional[Union[int, float]] = None,
                 val_begin: int = 1,
                 fp16: bool = False,
                 system='',
                 prompt_template=None,
                 max_new_tokens=256,
                 num_beams=1,
                 do_sample=True,
                 stop_word=None,
                 end_str=None,
                 val_interval: int = 1000,
                 dynamic_intervals: Optional[List[Tuple[int, int]]] = None
                 ) -> None:
        
        print(" " * 6 + "Inside TrainLoop init")
        print(" " * 7 + "↓")
        print(" " * 7 + "↓")
        
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
        self.fp16 = fp16
        self._max_epochs = 1  
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
        
        # for visualizing llm output during training
        if prompt_template is None:
            instruction = '{input}'
        else:
            if isinstance(prompt_template, str):  # for resume
                prompt_template = get_object_from_string(prompt_template)
            instruction = prompt_template.get('INSTRUCTION', '{input}')
            if system != '':
                system = prompt_template.get(
                    'SYSTEM', '{system}\n').format(system=system)
        self.instruction = instruction
        self.system = system
        self.max_new_tokens = max_new_tokens
        self.tokenizer = BUILDER.build(tokenizer)
        # default generation config
        self.gen_config = GenerationConfig(
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id,
        )
        if stop_word is not None:
            self.stop_criteria = get_stop_criteria(
                tokenizer=self.tokenizer, stop_words=stop_word)
        else:
            self.stop_criteria = StoppingCriteriaList()
        self.end_str = end_str

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
        
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook('after_train_iter', batch_idx=self._iter, data_batch=data_batch, outputs=outputs)
        
        
        self.runner.call_hook(
            'before_test_iter', batch_idx=self._iter, data_batch=data_batch)

        with torch.no_grad():
            if self.fp16:
                if is_model_wrapper(self.runner.model):
                    self.runner.model.module = self.runner.model.module.to(
                        torch.float16)
                else:
                    self.runner.model = self.runner.model.to(torch.float16)

            with autocast(enabled=self.fp16, dtype=torch.float16):
                if is_model_wrapper(self.runner.model):
                    data_preprocessor = self.runner.model.module.data_preprocessor
                    audio_data_dtype = self.runner.model.module.speech_encoder.encoder.conv1.weight.dtype
                    llm_data_dtype = self.runner.model.module.projector.model[0].weight.dtype
                    projector = self.runner.model.module.projector
                    speech_encoder = self.runner.model.module.speech_encoder
                    llm = self.runner.model.module.llm
                    generate_fn = self.runner.model.module.generate
                    decoder_start_token_id = self.runner.model.module.speech_encoder.config.decoder_start_token_id
                else:
                    data_preprocessor = self.runner.model.data_preprocessor
                    audio_data_dtype = self.runner.model.speech_encoder.encoder.conv1.weight.dtype
                    llm_data_dtype = self.runner.model.projector.model[0].weight.dtype
                    speech_encoder = self.runner.model.speech_encoder
                    projector = self.runner.model.projector
                    llm = self.runner.model.llm
                    generate_fn = self.runner.model.generate
                    decoder_start_token_id = self.runner.model.speech_encoder.config.decoder_start_token_id

                # Print the data batch
                print("Data Batch:", data_batch)

                data_batch = data_preprocessor(data_batch, False)
                data = data_batch['data']

                # Check data types for debugging
                print("Audio data dtype:", audio_data_dtype)
                print("LLM data dtype:", llm_data_dtype)

                data['audio_tokens'] = data['audio_tokens'].to(audio_data_dtype)
                batch_size = data['audio_tokens'].shape[0]
                decoder_input_ids = torch.tensor([[1] * batch_size]) * decoder_start_token_id

                # Encoder output
                audio_outputs = speech_encoder(
                    data['audio_tokens'],
                    decoder_input_ids=decoder_input_ids.to(
                        data['audio_tokens'].device),
                    output_hidden_states=True).encoder_last_hidden_state
                
                print("Audio Outputs Shape:", audio_outputs.shape)

                # Range checks
                print(f"Audio Encoder Output Range: Min = {audio_outputs.min()}, Max = {audio_outputs.max()}, Mean = {audio_outputs.mean()}")

                # Dtype checks
                print(f"Audio Encoder Output Dtype: {audio_outputs.dtype}")

                # Projector output
                audio_outputs = audio_outputs.to(llm_data_dtype)
                audio_outputs = audio_outputs[:, :max(data['audio_lens']), :]
                audio_tokens = projector(audio_outputs)
                data['audio_tokens'] = audio_tokens

                print("Audio Tokens Shape:", audio_tokens.shape)

                # NaN or Inf checks
                if torch.isnan(audio_tokens).any() or torch.isinf(audio_tokens).any():
                    print("Warning: Audio Projector output contains NaN or Inf values.")

                # Range checks
                print(f"Projector Output Range: Min = {audio_tokens.min()}, Max = {audio_tokens.max()}, Mean = {audio_tokens.mean()}")

                # Dtype checks
                print(f"Projector Output Dtype: {audio_tokens.dtype}")

                # Prepare inputs for LLM
                mm_inputs = prepare_inputs_labels_for_llast(
                    llm=llm,
                    input_ids=data['input_ids'],
                    audio_lens=data['audio_lens'],
                    audio_tokens=audio_tokens)

                print("MM Inputs:", mm_inputs)

                # Generation output
                mm_inputs['inputs_embeds'] = mm_inputs['inputs_embeds'].to(
                    llm_data_dtype)
                generation_output = generate_fn(
                    **mm_inputs,
                    max_new_tokens=self.max_new_tokens,
                    generation_config=self.gen_config,
                    bos_token_id=self.tokenizer.bos_token_id,
                    stopping_criteria=self.stop_criteria)

                print("Generation Output:", generation_output)

                # Decode predictions
                generations = self.tokenizer.batch_decode(
                    generation_output, skip_special_tokens=True)

                if self.end_str:
                    generations = [
                        item.split(self.end_str)[0] for item in generations
                    ]
                
                print("Generations:", generations)

            self.runner.call_hook('after_test_iter', batch_idx=self._iter, data_batch=data_batch, outputs=generations)
            self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the `val_interval`."""
        step = bisect.bisect(self.dynamic_milestones, (self._iter + 1))
        self.val_interval = self.dynamic_intervals[step - 1]

