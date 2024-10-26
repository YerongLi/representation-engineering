# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
import trace
from collections import OrderedDict
def trace_calls(frame, event, arg):
    if event != 'call':
        return
    co = frame.f_code
    func_name = co.co_name
    func_line_no = frame.f_lineno
    func_filename = co.co_filename
    with open('trace.txt', 'a') as f:
        f.write(f'Call to {func_name} on line {func_line_no} of {func_filename}\n')
    return trace_calls

def ensure_parameters_available(model):
    """
    Ensures that all parameters in the model are available for computation.
    If any parameter has a 'ds_status' attribute indicating it's not ready,
    CUDA synchronization is triggered with a 0.1-second wait.
    
    Args:
        model (torch.nn.Module): The model whose parameters are checked.
    """
    # Initial CUDA synchronization to ensure readiness
    torch.cuda.synchronize()

    # Check each parameter's status
    for _, param in model.named_parameters():
        # Only proceed if parameter has a 'ds_status' and is not available
        while hasattr(param, 'ds_status') and param.ds_status != "AVAILABLE":
            # print('sleep')
            time.sleep(0.1)  # Wait 0.1 seconds before retrying
            torch.cuda.synchronize()
                
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import gc
import pdb
import torch
from peft import PeftModel
from torch import nn
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

from swift.torchacc_utils import patch_clip_grad_norm, ta_trim_graph
from swift.utils import use_torchacc
# from .loss import get_loss_func
from swift.trainers.loss import get_loss_func
# from .mixin import SwiftMixin
from swift.trainers.mixin import SwiftMixin
# from .push_to_ms import PushToMsHubMixin
from swift.trainers.push_to_ms import PushToMsHubMixin

from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.exception import MaxLengthExceededError

_global_backward_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_backward_hooks: Dict[int, Callable] = OrderedDict()
_global_is_full_backward_hook: Optional[bool] = None
_global_forward_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_forward_hooks: Dict[int, Callable] = OrderedDict()
_global_forward_hooks_always_called: Dict[int, bool] = OrderedDict()

def process_model(self, *args, **kwargs):
    # if self.model: print(type(self.model))
    forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
    # If we don't have any hooks, we want to skip the rest of the logic in
    # this function, and just call forward.
    if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
            or _global_backward_pre_hooks or _global_backward_hooks
            or _global_forward_hooks or _global_forward_pre_hooks):
        return forward_call(*args, **kwargs)

    try:
        # print(args, kwargs)
        result = None
        called_always_called_hooks = set()

        full_backward_hooks, non_full_backward_hooks = [], []
        backward_pre_hooks = []
        if self._backward_pre_hooks or _global_backward_pre_hooks:
            backward_pre_hooks = self._get_backward_pre_hooks()

        if self._backward_hooks or _global_backward_hooks:
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()

        if _global_forward_pre_hooks or self._forward_pre_hooks:
            for hook_id, hook in (
                *_global_forward_pre_hooks.items(),
                *self._forward_pre_hooks.items(),
            ):
                if hook_id in self._forward_pre_hooks_with_kwargs:
                    args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
                    if args_kwargs_result is not None:
                        if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
                            args, kwargs = args_kwargs_result
                        else:
                            raise RuntimeError(
                                "forward pre-hook must return None or a tuple "
                                f"of (new_args, new_kwargs), but got {args_kwargs_result}."
                            )
                else:
                    args_result = hook(self, args)
                    if args_result is not None:
                        if not isinstance(args_result, tuple):
                            args_result = (args_result,)
                        args = args_result

        bw_hook = None
        if full_backward_hooks or backward_pre_hooks:
            bw_hook = hooks.BackwardHook(self, full_backward_hooks, backward_pre_hooks)
            args = bw_hook.setup_input_hook(args)
        return self
        # # result = forward_call(*args, **kwargs)
        # if _global_forward_hooks or self._forward_hooks:
        #     for hook_id, hook in (
        #         *_global_forward_hooks.items(),
        #         *self._forward_hooks.items(),
        #     ):
        #         # mark that always called hook is run
        #         if hook_id in self._forward_hooks_always_called or hook_id in _global_forward_hooks_always_called:
        #             called_always_called_hooks.add(hook_id)

        #         if hook_id in self._forward_hooks_with_kwargs:
        #             hook_result = hook(self, args, kwargs, result)
        #         else:
        #             hook_result = hook(self, args, result)

        #         if hook_result is not None:
        #             result = hook_result

        # if bw_hook:
        #     if not isinstance(result, (torch.Tensor, tuple)):
        #         warnings.warn("For backward hooks to be called,"
        #                       " module output should be a Tensor or a tuple of Tensors"
        #                       f" but received {type(result)}")
        #     result = bw_hook.setup_output_hook(result)

        # # Handle the non-full backward hooks
        # if non_full_backward_hooks:
        #     var = result
        #     while not isinstance(var, torch.Tensor):
        #         if isinstance(var, dict):
        #             var = next(v for v in var.values() if isinstance(v, torch.Tensor))
        #         else:
        #             var = var[0]
        #     grad_fn = var.grad_fn
        #     if grad_fn is not None:
        #         for hook in non_full_backward_hooks:
        #             grad_fn.register_hook(_WrappedHook(hook, self))
        #         self._maybe_warn_non_full_backward_hook(args, result, grad_fn)
        # return result

    except Exception:
        # run always called hooks if they have not already been run
        # For now only forward hooks have the always_call option but perhaps
        # this functionality should be added to full backward hooks as well.
        for hook_id, hook in _global_forward_hooks.items():
            if hook_id in _global_forward_hooks_always_called and hook_id not in called_always_called_hooks:  # type: ignore[possibly-undefined]
                try:
                    hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                    if hook_result is not None:
                        result = hook_result
                except Exception as e:
                    warnings.warn("global module forward hook with ``always_call=True`` raised an exception "
                                  f"that was silenced as another error was raised in forward: {str(e)}")
                    continue

        for hook_id, hook in self._forward_hooks.items():
            if hook_id in self._forward_hooks_always_called and hook_id not in called_always_called_hooks:  # type: ignore[possibly-undefined]
                try:
                    if hook_id in self._forward_hooks_with_kwargs:
                        hook_result = hook(self, args, kwargs, result)  # type: ignore[possibly-undefined]
                    else:
                        hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                    if hook_result is not None:
                        result = hook_result
                except Exception as e:
                    warnings.warn("module forward hook with ``always_call=True`` raised an exception "
                                  f"that was silenced as another error was raised in forward: {str(e)}")
                    continue
        # raise exception raised in try block
        raise


# DEBUG : use to_device in the utils
def to_device(inputs: Any, device: torch.device) -> Any:
    if callable(getattr(inputs, 'to', None)):
        return inputs.to(device=device)

    if isinstance(inputs, Mapping):
        res = {}
        for k, v in inputs.items():
            res[k] = to_device(v, device)
    elif isinstance(inputs, Sequence) and not isinstance(inputs, str):
        res = []
        for b in inputs:
            res.append(to_device(b, device))
    else:
        res = inputs
    return res


class Trainer(PushToMsHubMixin, SwiftMixin, HfTrainer):
    pass


class Seq2SeqTrainer(PushToMsHubMixin, SwiftMixin, HfSeq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # performance
        if not hasattr(self, 'perf'):
            self.perf = {}
        self.perf.update({
            'gen_time': 0.,
            'gen_len': 0,
        })
        self._acc = torch.tensor(0.).to(self.args.device)
        if use_torchacc():
            patch_clip_grad_norm(self.accelerator)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)

        inputs.pop('loss_scale', None)
        has_labels = 'labels' in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()

        if len(gen_kwargs) == 0 and hasattr(self, '_gen_kwargs'):
            gen_kwargs = self._gen_kwargs.copy()
            if hasattr(self.model, 'generation_config'):
                gen_kwargs.update(self.model.generation_config.to_dict())

        if gen_kwargs.get('max_length') is None and gen_kwargs.get('max_new_tokens') is None:
            gen_kwargs['max_length'] = self.model.config.max_length
        gen_kwargs['num_beams'] = (
            gen_kwargs['num_beams'] if gen_kwargs.get('num_beams') is not None else self.model.config.num_beams)
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs['synced_gpus'] = (
            gen_kwargs['synced_gpus'] if gen_kwargs.get('synced_gpus') is not None else default_synced_gpus)

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if ('labels' in inputs and 'decoder_input_ids' in inputs
                and inputs['labels'].shape == inputs['decoder_input_ids'].shape):
            inputs = {k: v for k, v in inputs.items() if k != 'decoder_input_ids'}

        gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        gen_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        # fix generate warning
        if 'max_length' in gen_kwargs and 'max_new_tokens' in gen_kwargs and gen_kwargs['max_new_tokens'] is not None:
            gen_kwargs.pop('max_length')
        gen_time = time.time()
        generate_inputs = inputs.copy()
        if has_labels:
            _labels = inputs['labels'][0]
            n_mask = 0
            for i in range(len(_labels)):
                if _labels[i] != -100:
                    n_mask = i
                    break

            for k in ['input_ids', 'attention_mask']:
                generate_inputs[k] = generate_inputs[k][:, :n_mask]
            generate_inputs['labels'] = generate_inputs['labels'][:, n_mask:]

        generated_tokens = self.model.generate(**generate_inputs, **gen_kwargs)
        gen_time = time.time() - gen_time

        if hasattr(self.model, 'encoder') and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = generate_inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = generate_inputs[self.model.main_input_name]

        generated_tokens = generated_tokens[:, generation_inputs.shape[1]:]
        gen_len = len(generated_tokens[0])
        self.perf['gen_time'] = self.perf['gen_time'] + gen_time
        self.perf['gen_len'] = self.perf['gen_len'] + gen_len

        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get('max_length') is not None and generated_tokens.shape[-1] < gen_kwargs['max_length']:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs['max_length'])
        elif gen_kwargs.get('max_new_tokens') is not None and generated_tokens.shape[-1] < (gen_kwargs['max_new_tokens']
                                                                                            + 1):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs['max_new_tokens'] + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs['labels']).mean().detach()
                else:
                    loss = (outputs['loss'] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = generate_inputs['labels']
            if gen_kwargs.get('max_length') is not None and labels.shape[-1] < gen_kwargs['max_length']:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs['max_length'])
            elif gen_kwargs.get('max_new_tokens') is not None and labels.shape[-1] < (gen_kwargs['max_new_tokens'] + 1):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs['max_new_tokens'] + 1))
        else:
            labels = None

        return loss, generated_tokens, labels

    def compute_loss(self, model, inputs, return_outputs=None):
        if not hasattr(self, '_custom_metrics'):
            self._custom_metrics = {}

        labels = None
        loss_name = self.args.loss_name
        if loss_name is None and 'loss_scale' in inputs:
            loss_name = 'loss-scale'

        loss_kwargs = {}
        if loss_name == 'loss-scale':
            loss_kwargs['loss_scale'] = inputs.pop('loss_scale')

        if loss_name is not None or self.label_smoother is not None and 'labels' in inputs:
            labels = inputs.pop('labels')

        loss_kwargs['labels'] = labels
        outputs = model(**inputs)
        if loss_name is not None:
            loss_func = get_loss_func(loss_name)
            outputs['loss'] = loss_func(outputs, **loss_kwargs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None and loss_name is None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

        if labels is None:
            labels = inputs['labels']

        if self.sequence_parallel_size > 1:
            from swift.trainers.xtuner import reduce_xtuner_sequence_parallel_loss
            loss = reduce_xtuner_sequence_parallel_loss(loss, labels)

        if self.is_encoder_decoder:
            preds = outputs.logits.argmax(dim=2)[..., :] if outputs.logits is not None else None
            labels = labels[..., :]
        else:
            preds = outputs.logits.argmax(dim=2)[..., :-1] if outputs.logits is not None else None
            labels = labels[..., 1:]

        masks = labels != -100
        acc_strategy = getattr(self.args, 'acc_strategy', 'token')
        acc: Optional[torch.Tensor] = None
        sft_args = getattr(self, 'sft_args', None)
        acc_steps = 1 if sft_args is None else sft_args.acc_steps
        if self.state.global_step % acc_steps == 0 and preds is not None:
            if preds.shape != labels.shape:
                pass
            elif acc_strategy == 'sentence':
                acc_list = []
                for i, m in enumerate(masks):
                    acc_list.append(torch.all(preds[i, m] == labels[i, m]).to(torch.int64).item())
                acc = torch.tensor(acc_list, device=preds.device).float().mean()
            else:
                if use_torchacc():
                    ta_trim_graph()
                    preds = preds.to('cpu')
                    masks = masks.to('cpu')
                    labels = labels.to('cpu')
                acc = (torch.masked_select(preds, masks) == torch.masked_select(labels, masks)).float().mean()
            if model.training and acc is not None:
                if 'acc' not in self._custom_metrics:
                    self._custom_metrics['acc'] = self._acc
                self._custom_metrics['acc'] = self._custom_metrics['acc'] + acc / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss


class RETrainer(PushToMsHubMixin, SwiftMixin, HfSeq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # performance
        if not hasattr(self, 'perf'):
            self.perf = {}
        self.perf.update({
            'gen_time': 0.,
            'gen_len': 0,
        })
        self._acc = torch.tensor(0.).to(self.args.device)
        if use_torchacc():
            patch_clip_grad_norm(self.accelerator)
        self.template = None
        self.alpha = None
        self.target_layers = None
        self.pre_loss = None

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)

        inputs.pop('loss_scale', None)
        has_labels = 'labels' in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()

        if len(gen_kwargs) == 0 and hasattr(self, '_gen_kwargs'):
            gen_kwargs = self._gen_kwargs.copy()
            if hasattr(self.model, 'generation_config'):
                gen_kwargs.update(self.model.generation_config.to_dict())

        if gen_kwargs.get('max_length') is None and gen_kwargs.get('max_new_tokens') is None:
            gen_kwargs['max_length'] = self.model.config.max_length
        gen_kwargs['num_beams'] = (
            gen_kwargs['num_beams'] if gen_kwargs.get('num_beams') is not None else self.model.config.num_beams)
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs['synced_gpus'] = (
            gen_kwargs['synced_gpus'] if gen_kwargs.get('synced_gpus') is not None else default_synced_gpus)

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if ('labels' in inputs and 'decoder_input_ids' in inputs
                and inputs['labels'].shape == inputs['decoder_input_ids'].shape):
            inputs = {k: v for k, v in inputs.items() if k != 'decoder_input_ids'}

        gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        gen_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        # fix generate warning
        if 'max_length' in gen_kwargs and 'max_new_tokens' in gen_kwargs and gen_kwargs['max_new_tokens'] is not None:
            gen_kwargs.pop('max_length')
        gen_time = time.time()
        generate_inputs = inputs.copy()
        if has_labels:
            _labels = inputs['labels'][0]
            n_mask = 0
            for i in range(len(_labels)):
                if _labels[i] != -100:
                    n_mask = i
                    break

            for k in ['input_ids', 'attention_mask']:
                generate_inputs[k] = generate_inputs[k][:, :n_mask]
            generate_inputs['labels'] = generate_inputs['labels'][:, n_mask:]

        generated_tokens = self.model.generate(**generate_inputs, **gen_kwargs)
        gen_time = time.time() - gen_time

        if hasattr(self.model, 'encoder') and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = generate_inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = generate_inputs[self.model.main_input_name]

        generated_tokens = generated_tokens[:, generation_inputs.shape[1]:]
        gen_len = len(generated_tokens[0])
        self.perf['gen_time'] = self.perf['gen_time'] + gen_time
        self.perf['gen_len'] = self.perf['gen_len'] + gen_len

        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get('max_length') is not None and generated_tokens.shape[-1] < gen_kwargs['max_length']:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs['max_length'])
        elif gen_kwargs.get('max_new_tokens') is not None and generated_tokens.shape[-1] < (gen_kwargs['max_new_tokens']
                                                                                            + 1):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs['max_new_tokens'] + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs['labels']).mean().detach()
                else:
                    loss = (outputs['loss'] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = generate_inputs['labels']
            if gen_kwargs.get('max_length') is not None and labels.shape[-1] < gen_kwargs['max_length']:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs['max_length'])
            elif gen_kwargs.get('max_new_tokens') is not None and labels.shape[-1] < (gen_kwargs['max_new_tokens'] + 1):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs['max_new_tokens'] + 1))
        else:
            labels = None

        return loss, generated_tokens, labels
    

    def compute_loss(self, model, inputs, return_outputs=None):
        if not hasattr(self, '_custom_metrics'):
            self._custom_metrics = {}
        # module = process_model(model.module.model, (), inputs[0])
        module = model.module
        labels = None
        loss_name = self.args.loss_name
        if loss_name is None and 'loss_scale' in inputs:
            loss_name = 'loss-scale'
        
        loss_kwargs = {}
        if loss_name == 'loss-scale':
            loss_kwargs['loss_scale'] = inputs.pop('loss_scale')

        if loss_name is not None or self.label_smoother is not None and 'labels' in inputs:
            labels = inputs.pop('labels')

        loss_kwargs['labels'] = labels
        try:
            concatenated_inputs = [self.template.conkat(inputs[i], module) for i in range(3)]
        except MaxLengthExceededError:
            if not self.pre_loss:
                self.pre_loss = torch.tensor(0.0, requires_grad=True).to(model.device)
            return self.pre_loss
        response_attention_mask = concatenated_inputs[0]['attention_mask'][:, -self.template.response_max_len:].repeat(len(self.target_layers), 1, 1).unsqueeze(-1)
        # print(concatenated_inputs[0].pop('inputs_embeds'))
        module = 'past_key_values' # 'hidden_states
        with model.disable_adapter():
            model.eval()
            with torch.no_grad():
                orig_outputs = model(
                    **(concatenated_inputs[0]),
                    output_hidden_states=True
                )['hidden_states']
                orig_hidden = [orig_outputs[l][:, -self.template.response_max_len:].detach() for l in self.target_layers]
                
                pos_outputs = model(
                    **(concatenated_inputs[1]),
                    output_hidden_states=True
                )['hidden_states']

                neg_outputs = model(
                    **(concatenated_inputs[2]),
                    output_hidden_states=True
                )['hidden_states']

                direction_hidden = [pos_outputs[l][:, -self.template.response_max_len:].detach() - \
                                    neg_outputs[l][:, -self.template.response_max_len:].detach() \
                                    # + beta * torch.tensor(pca_directions[l - len(pca_directions)], device=model.device, dtype=torch.float16) \
                                                    for l in self.target_layers]
                target_hidden = torch.stack([orig_hidden[i] + self.alpha * direction_hidden[i] for i in range(len(self.target_layers))]) * response_attention_mask
                del orig_outputs, pos_outputs, neg_outputs, orig_hidden, direction_hidden
                gc.collect()
                torch.cuda.empty_cache()

        model.train()
        lora_outputs = model(
            **(concatenated_inputs[0]),
            output_hidden_states=True
        )['hidden_states']
        lora_hidden = torch.stack([lora_outputs[l][:, -self.template.response_max_len:] for l in self.target_layers]) * response_attention_mask

        loss_fct = torch.nn.MSELoss()
        loss = torch.norm(lora_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
        self.pre_loss = loss
        return (loss, lora_hidden) if return_outputs else loss

    
    def evaluate(self, eval_dataset=None, ignore_keys=None, sanity_check=False, **kwargs):
        # print(eval_dataset)
        print(f"Query Max Length: {self.template.query_max_len}")
        print(f"Response Max Length: {self.template.response_max_len}")
        print(f"MODEL Max Length: {self.model.max_length}")