# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import os
import time
import shutil

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader
from torch.distributed.checkpoint import save_state_dict, load_state_dict
from safetensors.torch import load_file, save_file

from modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
    )


def _save_with_retry(save_fn, filepath, max_retries=3, retry_delay=5, logger=None):
    """Save with retry logic for handling transient I/O errors."""
    for attempt in range(max_retries):
        try:
            save_fn()
            # Verify file exists and has non-zero size
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                return True
            else:
                raise IOError(f"File {filepath} was not written correctly")
        except Exception as e:
            if logger:
                logger.warning(f"Save attempt {attempt + 1}/{max_retries} failed for {filepath}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                raise
    return False


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir,
        train_steps,
        model,
        ema_model,
        optimizer,
        scheduler,
        data_status,
        logger,
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        temp_path = save_path + ".tmp"
        rank = dist.get_rank()

        try:
            # Create temp directory
            if rank == 0:
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path, ignore_errors=True)
                os.makedirs(temp_path, exist_ok=True)
            dist.barrier()

            # Ensure temp_path is visible to all ranks
            os.makedirs(temp_path, exist_ok=True)

            logger.info(f"Saving checkpoint to {save_path} (using temp: {temp_path}).")

            # Clear GPU cache before saving to free up memory for FSDP gather
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Save EMA model
            if ema_model is not None:
                with FSDP.state_dict_type(
                    ema_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    ema_state_dict = ema_model.state_dict()
                    if rank == 0:
                        ema_path = os.path.join(temp_path, "ema.safetensors")
                        _save_with_retry(
                            lambda: save_file(ema_state_dict, ema_path),
                            ema_path,
                            logger=logger
                        )
                        logger.info(f"EMA model saved: {os.path.getsize(ema_path) / 1e9:.2f} GB")
                    del ema_state_dict
                torch.cuda.empty_cache()

            # Barrier after EMA save to ensure rank 0 is done
            dist.barrier()

            # Save model
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                model_state_dict = model.state_dict()
                if rank == 0:
                    model_path = os.path.join(temp_path, "model.safetensors")
                    _save_with_retry(
                        lambda: save_file(model_state_dict, model_path),
                        model_path,
                        logger=logger
                    )
                    logger.info(f"Model saved: {os.path.getsize(model_path) / 1e9:.2f} GB")
                del model_state_dict
            torch.cuda.empty_cache()

            # Barrier after model save to ensure rank 0 is done before optimizer saving
            dist.barrier()

            # Save optimizer shards
            with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
                if fsdp_config.sharding_strategy == "FULL_SHARD":
                    shard_index = rank
                    total_shards = dist.get_world_size()
                elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                    shard_index = rank % fsdp_config.num_shard
                    total_shards = fsdp_config.num_shard
                else:
                    raise NotImplementedError

                optimizer_save_path = os.path.join(
                    temp_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
                )

                should_save = False
                if fsdp_config.sharding_strategy == "FULL_SHARD":
                    should_save = True
                elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                    should_save = rank < fsdp_config.num_shard

                if should_save:
                    opt_state = optimizer.state_dict()
                    _save_with_retry(
                        lambda: torch.save(opt_state, optimizer_save_path),
                        optimizer_save_path,
                        logger=logger
                    )
                    del opt_state

            # Barrier after optimizer save
            dist.barrier()

            # Save scheduler and data_status (rank 0 only)
            if rank == 0:
                if scheduler is not None:
                    scheduler_path = os.path.join(temp_path, "scheduler.pt")
                    _save_with_retry(
                        lambda: torch.save(scheduler.state_dict(), scheduler_path),
                        scheduler_path,
                        logger=logger
                    )

                if data_status is not None:
                    data_status_path = os.path.join(temp_path, "data_status.pt")
                    _save_with_retry(
                        lambda: torch.save(data_status, data_status_path),
                        data_status_path,
                        logger=logger
                    )

            dist.barrier()

            # Atomic rename: only if everything succeeded
            if rank == 0:
                if os.path.exists(save_path):
                    # Remove old checkpoint if exists
                    shutil.rmtree(save_path, ignore_errors=True)
                os.rename(temp_path, save_path)
                logger.info(f"Checkpoint saved successfully to {save_path}.")

            dist.barrier()
            return

        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {train_steps}: {e}")
            # Cleanup temp directory on failure
            if rank == 0 and os.path.exists(temp_path):
                shutil.rmtree(temp_path, ignore_errors=True)
                logger.info(f"Cleaned up incomplete checkpoint at {temp_path}")
            dist.barrier()
            # Re-raise to let training loop handle it (could continue training)
            raise

    @staticmethod
    def fsdp_save_fsdp_ckpt(
        ckpt_dir,
        train_steps,
        model,
        ema_model,
        optimizer,
        scheduler,
        data_status,
        logger,
        fsdp_config,
    ):
        """
        Save checkpoint using SHARDED_STATE_DICT to avoid OOM.
        Each GPU saves its own shard, no need to gather all weights to rank 0.
        Reference: https://github.com/ByteDance-Seed/Bagel/issues/139
        """
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        temp_path = save_path + ".tmp"
        rank = dist.get_rank()

        try:
            # Create temp directory
            if rank == 0:
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path, ignore_errors=True)
                os.makedirs(temp_path, exist_ok=True)
            dist.barrier()

            # Ensure temp_path is visible to all ranks
            os.makedirs(temp_path, exist_ok=True)

            logger.info(f"Saving sharded checkpoint to {save_path} (using temp: {temp_path}).")

            # Clear GPU cache before saving
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Save EMA model using SHARDED_STATE_DICT
            if ema_model is not None:
                with FSDP.state_dict_type(
                    ema_model,
                    StateDictType.SHARDED_STATE_DICT,
                    ShardedStateDictConfig(offload_to_cpu=True),
                ):
                    ema_state_dict = ema_model.state_dict()
                    ema_save_dir = os.path.join(temp_path, "ema")
                    ema_writer = FileSystemWriter(ema_save_dir)
                    save_state_dict(ema_state_dict, ema_writer)
                    del ema_state_dict
                gc.collect()
                torch.cuda.empty_cache()
                if rank == 0:
                    logger.info(f"EMA model saved to {ema_save_dir}")

            dist.barrier()

            # Save model using SHARDED_STATE_DICT
            with FSDP.state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                model_state_dict = model.state_dict()
                model_save_dir = os.path.join(temp_path, "model")
                model_writer = FileSystemWriter(model_save_dir)
                save_state_dict(model_state_dict, model_writer)
                del model_state_dict
            gc.collect()
            torch.cuda.empty_cache()
            if rank == 0:
                logger.info(f"Model saved to {model_save_dir}")

            dist.barrier()

            # Save optimizer shards (same as before)
            with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
                if fsdp_config.sharding_strategy == "FULL_SHARD":
                    shard_index = rank
                    total_shards = dist.get_world_size()
                elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                    shard_index = rank % fsdp_config.num_shard
                    total_shards = fsdp_config.num_shard
                else:
                    raise NotImplementedError

                optimizer_save_path = os.path.join(
                    temp_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
                )

                should_save = False
                if fsdp_config.sharding_strategy == "FULL_SHARD":
                    should_save = True
                elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                    should_save = rank < fsdp_config.num_shard

                if should_save:
                    opt_state = optimizer.state_dict()
                    _save_with_retry(
                        lambda: torch.save(opt_state, optimizer_save_path),
                        optimizer_save_path,
                        logger=logger
                    )
                    del opt_state

            dist.barrier()

            # Save scheduler (rank 0 only)
            if rank == 0:
                if scheduler is not None:
                    scheduler_path = os.path.join(temp_path, "scheduler.pt")
                    _save_with_retry(
                        lambda: torch.save(scheduler.state_dict(), scheduler_path),
                        scheduler_path,
                        logger=logger
                    )

            # Save data_status per rank
            if data_status is not None:
                data_status_dir = os.path.join(temp_path, "data_status")
                os.makedirs(data_status_dir, exist_ok=True)
                data_status_path = os.path.join(data_status_dir, f"rank{rank:05d}.pt")
                _save_with_retry(
                    lambda: torch.save(data_status, data_status_path),
                    data_status_path,
                    logger=logger
                )

            dist.barrier()

            # Atomic rename: only if everything succeeded
            if rank == 0:
                if os.path.exists(save_path):
                    shutil.rmtree(save_path, ignore_errors=True)
                os.rename(temp_path, save_path)
                logger.info(f"Sharded checkpoint saved successfully to {save_path}.")

            dist.barrier()
            return

        except Exception as e:
            logger.error(f"Failed to save sharded checkpoint at step {train_steps}: {e}")
            if rank == 0 and os.path.exists(temp_path):
                shutil.rmtree(temp_path, ignore_errors=True)
                logger.info(f"Cleaned up incomplete checkpoint at {temp_path}")
            dist.barrier()
            raise

    @staticmethod
    def try_load_fsdp_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        """
        Load checkpoint with support for both old (safetensors) and new (sharded) formats.

        Old format: model.safetensors, ema.safetensors (single files)
        New format: model/, ema/ directories with .distcp shards

        IMPORTANT: For new format (sharded), model must already be wrapped in FSDP before calling this.
        """
        if resume_from is None or not os.path.exists(resume_from):
            logger.info(f"No checkpoint found at {resume_from}, training from scratch.")
            return model, ema_model

        logger.info(f"Loading checkpoint from {resume_from}.")

        # Detect format: check for model/ directory (new) vs model.safetensors (old)
        model_dir = os.path.join(resume_from, "model")
        model_safetensors = os.path.join(resume_from, "model.safetensors")
        ema_dir = os.path.join(resume_from, "ema")
        ema_safetensors = os.path.join(resume_from, "ema.safetensors")

        is_sharded_format = os.path.isdir(model_dir)

        if is_sharded_format:
            # New sharded format - use FileSystemReader
            logger.info(f"Detected sharded checkpoint format.")

            # Determine which model to load
            if resume_from_ema and os.path.isdir(ema_dir):
                model_load_dir = ema_dir
                logger.info(f"Loading model from EMA: {model_load_dir}")
            else:
                model_load_dir = model_dir
                logger.info(f"Loading model from: {model_load_dir}")

            # Load model using SHARDED_STATE_DICT
            with FSDP.state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                model_state_dict = model.state_dict()
                model_reader = FileSystemReader(model_load_dir)
                load_state_dict(model_state_dict, model_reader)
                model.load_state_dict(model_state_dict, strict=False)
                del model_state_dict
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"Model loaded from sharded checkpoint.")

            # Load EMA model if provided
            if ema_model is not None:
                if os.path.isdir(ema_dir):
                    ema_load_dir = ema_dir
                else:
                    # Fall back to model if ema doesn't exist
                    logger.info(f"EMA directory not found, replicating from model.")
                    ema_load_dir = model_load_dir

                with FSDP.state_dict_type(
                    ema_model,
                    StateDictType.SHARDED_STATE_DICT,
                    ShardedStateDictConfig(offload_to_cpu=True),
                ):
                    ema_state_dict = ema_model.state_dict()
                    ema_reader = FileSystemReader(ema_load_dir)
                    load_state_dict(ema_state_dict, ema_reader)
                    ema_model.load_state_dict(ema_state_dict, strict=False)
                    del ema_state_dict
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"EMA model loaded from sharded checkpoint.")

        else:
            # Old safetensors format - use load_file
            logger.info(f"Detected safetensors checkpoint format.")

            if resume_from_ema:
                model_state_dict_path = ema_safetensors
            else:
                model_state_dict_path = model_safetensors

            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off
            model_state_dict.pop('latent_pos_embed.pos_embed', None)
            model_state_dict.pop('vit_pos_embed.pos_embed', None)
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                if not os.path.exists(ema_safetensors):
                    logger.info(f"Replicating EMA model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                else:
                    ema_state_dict_path = ema_safetensors
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                ema_state_dict.pop('latent_pos_embed.pos_embed', None)
                ema_state_dict.pop('vit_pos_embed.pos_embed', None)
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict

        return model, ema_model

    @staticmethod
    def try_load_fsdp_train_state(resume_from, optimizer, scheduler, fsdp_config):
        """
        Load training state with support for both old and new checkpoint formats.
        """
        if resume_from is None or not os.path.exists(resume_from):
            return optimizer, scheduler, 0, None

        rank = dist.get_rank()

        if fsdp_config.sharding_strategy == "FULL_SHARD":
            shard_index = rank
            total_shards = dist.get_world_size()
        elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
            shard_index = rank % fsdp_config.num_shard
            total_shards = fsdp_config.num_shard
        else:
            raise NotImplementedError

        # Load optimizer
        optimizer_state_dict_path = os.path.join(
            resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
        )
        optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict

        # Load scheduler
        scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
        scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
        scheduler.load_state_dict(scheduler_state_dict)
        del scheduler_state_dict

        train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1

        # Load data_status - check for new per-rank format first
        data_status_dir = os.path.join(resume_from, "data_status")
        data_status_path_new = os.path.join(data_status_dir, f"rank{rank:05d}.pt")
        data_status_path_old = os.path.join(resume_from, "data_status.pt")

        data_status = None
        if os.path.exists(data_status_path_new):
            # New format: per-rank data_status
            data_status = torch.load(data_status_path_new, weights_only=True, map_location="cpu")
        elif os.path.exists(data_status_path_old):
            # Old format: single data_status.pt with list
            data_status_list = torch.load(data_status_path_old, weights_only=True, map_location="cpu")
            if rank < len(data_status_list):
                data_status = data_status_list[rank]

        return optimizer, scheduler, train_steps, data_status

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            model_state_dict.pop('latent_pos_embed.pos_embed')
            model_state_dict.pop('vit_pos_embed.pos_embed')
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                ema_state_dict.pop('latent_pos_embed.pos_embed')
                ema_state_dict.pop('vit_pos_embed.pos_embed')
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
