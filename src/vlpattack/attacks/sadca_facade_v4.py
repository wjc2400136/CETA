from __future__ import annotations


import copy


import time


from .sadca_adapter_v4 import attack_batch4, configure_variant, _verify


class SADCABatch4Facade:
    implementation_variant = 'server_sadca_alignment5_dynamic2x5_native_b4_facade_v27_native4_owner_model_microbatch_v4'

    def __init__(self, core, *, baseline_path, trace_hook=None):
        if trace_hook is not None and not callable(trace_hook):
            raise TypeError('trace_hook must be callable or None')
        self.core = core
        self.baseline_path = baseline_path
        self.trace_hook = trace_hook
        if not hasattr(core, '_sadca_batch_variant_receipt'):
            configure_variant(core, 5, baseline_path=baseline_path)
        else:
            _verify(core, baseline_path)
            receipt = core._sadca_batch_variant_receipt
            if receipt['actual_inner_steps'] != 5 or receipt['dynamic_rounds'] != 2:
                raise ValueError('Only the explicit alignment5 + dynamic2x5 profile is admitted')
        self.canonical_B1_implementation_variant = core.implementation_variant
        self.native_attack_images_per_call = 4  # Admitted maximum; actual N is recorded per call.
        self._clear()

    def __getattr__(self, name):
        core = self.__dict__.get('core')
        if core is None:
            raise AttributeError(name)
        return getattr(core, name)

    def _clear(self):
        self.last_method_observation = None
        self.last_batch4_observation = None
        self.last_final_source_word_witnesses = None
        self.last_final_source_word_witnesses_by_owner = None
        self.last_task_bindings = None
        self.last_task_binding = None
        self.last_facade_wall_seconds_without_outer_sync = None
        # The old B1 interval excludes negative preparation. The B4 adapter's
        # inclusive wall is a different measured interval, kept under its own key.
        self.last_native_attack_seconds = None
        self.last_call_state = 'NOT_STARTED'

    @staticmethod
    def _grouped(images, texts, lengths):
        if isinstance(texts, (str, bytes)) or isinstance(lengths, (str, bytes)):
            raise ValueError('Flat ordered captions and explicit owner lengths are required')
        flat = list(texts)
        sizes = tuple(lengths)
        if (not 1 <= len(sizes) <= 4
                or any(type(value) is not int or value < 1 for value in sizes)
                or sum(sizes) != len(flat)
                or any(not isinstance(value, str) for value in flat)
                or getattr(images, 'ndim', None) != 4 or len(images) != len(sizes)):
            raise ValueError('Require N=1..4 source images and complete ragged caption groups')
        groups = []
        offset = 0
        for size in sizes:
            groups.append(flat[offset:offset+size])
            offset += size
        return flat, sizes, groups

    def _return(self, images, lengths, pixels, groups, observation, started):
        if pixels.shape != images.shape or len(groups) != len(lengths):
            raise ValueError('SADCA B4 omitted an annotation or changed its source carrier')
        witnesses = observation['final_source_word_witnesses_by_owner']
        if len(witnesses) != len(lengths):
            raise ValueError('SADCA final-round word witnesses omitted an annotation')
        flat = []
        flat_witnesses = []
        for size, group, raw in zip(lengths, groups, witnesses, strict=True):
            if (len(group) != size or len(raw) != size
                    or any(not isinstance(text, str) for text in group)
                    or any(not isinstance(witness, dict) for witness in raw)):
                raise ValueError('SADCA changed a complete positive-caption group or witness count')
            flat.extend(group)
            # Original SADCA witnesses are ordered rows without caption_index.
            # Preserve each actual native record unchanged; the list order is
            # the batch-flat caption coordinate, and owner-local lists survive.
            flat_witnesses.extend(copy.deepcopy(raw))
        public = copy.deepcopy(observation)
        public.update(
            facade_variant=self.implementation_variant,
            canonical_B1_implementation_variant=self.canonical_B1_implementation_variant,
            text_group_lens=list(lengths),
            actual_native_attack_images_per_call=len(lengths),
            native_B1_seconds_interval_not_measured=True,
            full_generation_interval_owned_by_outer_caller=True,
        )
        self.last_batch4_observation = observation
        self.last_method_observation = public
        self.last_final_source_word_witnesses = flat_witnesses
        self.last_final_source_word_witnesses_by_owner = copy.deepcopy(witnesses)
        self.last_task_bindings = copy.deepcopy(observation['task_bindings'])
        self.last_facade_wall_seconds_without_outer_sync = time.perf_counter() - started
        self.last_call_state = 'RETURNED_PENDING_CALLER_VALIDATION'
        return pixels, flat

    def attack(self, images, texts, text_group_lens, *, source_negative_pool,
               image_ids, text_id_groups, owner_ids=None, trace_hook=None):
        self._clear()
        self.last_call_state = 'STARTED'
        started = time.perf_counter()
        try:
            _flat, lengths, groups = self._grouped(images, texts, text_group_lens)
            hook = self.trace_hook if trace_hook is None else trace_hook
            if hook is not None and not callable(hook):
                raise TypeError('trace_hook must be callable or None')
            pixels, adverse, observation = attack_batch4(
                self.core, images, groups, image_ids, text_id_groups,
                source_negative_pool, hook, baseline_path=self.baseline_path,
                owner_ids=owner_ids,
            )
            return self._return(images, lengths, pixels, adverse, observation, started)
        except BaseException:
            self.last_call_state = 'FAILED_PROPAGATED_NO_RETRY'
            raise

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)
