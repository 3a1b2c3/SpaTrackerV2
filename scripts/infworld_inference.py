"""
Infinite World - Action-Conditioned Video Generation Inference Script
======================================================================
A standalone inference script for generating long videos with action control.
"""

import argparse
import sys
import os
import cv2
import math
import torch
import random
import json
import datetime
import importlib
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch.distributed as dist
import torchvision.transforms as transforms
import re

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from infworld.utils.prepare_dataloader import get_obj_from_str
from infworld.utils.data_utils import get_first_clip_from_video, save_silent_video
from infworld.utils.dataset_utils import is_vid, is_img

# ============================================================================
# Action Mapping Dictionaries
# ============================================================================
MOVE_ACTION_MAP = {
    'no-op': 0,
    'go forward': 1,
    'go back': 2,
    'go left': 3,
    'go right': 4,
    'go forward and go left': 5,
    'go forward and go right': 6,
    'go back and go left': 7,
    'go back and go right': 8,
    'uncertain': 9
}

VIEW_ACTION_MAP = {
    'no-op': 0,
    'turn up': 1,
    'turn down': 2,
    'turn left': 3,
    'turn right': 4,
    'turn up and turn left': 5,
    'turn up and turn right': 6,
    'turn down and turn left': 7,
    'turn down and turn right': 8,
    'uncertain': 9
}

# ============================================================================
# Utility Functions
# ============================================================================
def extract_ckpt_step(path):
    """Extract checkpoint step number from path."""
    match = re.search(r'checkpoint-(\d+)\.ckpt', path)
    return int(match.group(1)) if match else 0

def resize_and_center_crop(image, target_size):
    """Resize image and center crop to target size."""
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = max(target_h / orig_h, target_w / orig_w)
    final_h = math.ceil(scale * orig_h)
    final_w = math.ceil(scale * orig_w)
    
    resized = cv2.resize(image, (final_w, final_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized)[None, ...].permute(0, 3, 1, 2).contiguous()
    cropped = transforms.functional.center_crop(tensor, target_size)
    return cropped[:, :, None, :, :]  # [1, C, 1, H, W]

def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def torch_gc():
    """Clear GPU memory cache."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def load_action_sequence(action_path):
    """Load action sequence from JSON file."""
    with open(action_path, 'r') as f:
        actions = json.load(f)
    
    move_indices = [MOVE_ACTION_MAP[a['move']] for a in actions]
    view_indices = [VIEW_ACTION_MAP[a['view']] for a in actions]
    return move_indices, view_indices

def load_condition_image(image_path, bucket_config):
    """Load and preprocess condition image."""
    if is_vid(image_path):
        frames = get_first_clip_from_video(image_path, clip_len=1)
    elif is_img(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames = [image]
    else:
        raise ValueError(f'Unsupported file format: {image_path}')
    
    processed_frames = []
    for frame in frames:
        ratio = frame.shape[0] / frame.shape[1]
        closest_bucket = sorted(bucket_config.keys(), key=lambda x: abs(float(x) - ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        
        tensor = resize_and_center_crop(frame, (target_h, target_w))
        tensor = (tensor / 255 - 0.5) * 2  # Normalize to [-1, 1]
        processed_frames.append(tensor)
    
    return torch.cat(processed_frames, dim=2)

# ============================================================================
# Distributed Setup (support single-GPU without torchrun to avoid port conflict)
# ============================================================================
def setup_distributed():
    """Setup distributed or single-GPU mode."""
    if 'RANK' in os.environ:
        # Launched by torchrun or similar
        rank = int(os.environ['RANK'])
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
        global_rank = dist.get_rank()
        num_processes = dist.get_world_size()
        return local_rank, global_rank, num_processes, True  # use_cp_init=True
    else:
        # Single process (no torchrun) - avoid port conflict, no dist init
        local_rank = 0
        global_rank = 0
        num_processes = 1
        torch.cuda.set_device(local_rank)
        return local_rank, global_rank, num_processes, False  # use_cp_init=False

local_rank, global_rank, num_processes, use_dist = setup_distributed()
print(f"[InfWorld] local_rank: {local_rank} | global_rank: {global_rank} | world_size: {num_processes}")

# Context parallel setup
context_parallel_size = 1
import infworld.context_parallel.context_parallel_util as cp_util
if use_dist:
    from infworld.context_parallel.context_parallel_util import init_context_parallel, get_dp_size, get_dp_rank
    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    dp_rank = get_dp_rank()
    dp_size = get_dp_size()
else:
    # Single process: set globals so get_dp_rank/get_dp_size work without dist
    cp_util.dp_rank = 0
    cp_util.dp_size = 1
    cp_util.cp_rank = 0
    cp_util.cp_size = 1
    dp_rank = 0
    dp_size = 1
enable_context_parallel = (context_parallel_size > 1)

# ============================================================================
# Configuration
# ============================================================================
# Inference settings
GLOBAL_SEED = 42
setup_seed(GLOBAL_SEED + global_rank)

TEXT_CFG_SCALE = 5.0
NUM_SAMPLING_STEPS = 5
SHIFT = 7  # PX256: 3, PX627: 7, PX960: 11
NUM_CHUNKS = 2  # Number of video chunks to generate
HIGH_QUALITY_SAVE = True

# Paths - checkpoint_path is read from config (configs/infworld_config.yaml)
# Model config - use standalone config
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'infworld_config.yaml')

PROMPTS_YAML = os.path.join(PROJECT_ROOT, 'prompts', 'demo.yaml')
BUCKET_CONFIG_NAME = 'ASPECT_RATIO_627_F64'

# Output directory
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'outputs')

# Negative prompt for generation quality
NEGATIVE_PROMPT = "many cars, crowds, Vivid hues, overexposed, static, blurry details, subtitles, style, work, artwork, image, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, motionless image, cluttered background, three legs, crowded background, walking backwards."

# ============================================================================
# Main Inference Loop
# ============================================================================
def resolve_path(path, root=PROJECT_ROOT):
    """Resolve path: if relative, join with project root."""
    if path is None:
        return path
    path = str(path).strip()
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    return path


def load_dit_state_dict(checkpoint_path):
    """Load DiT state dict from .ckpt (torch) or .safetensors."""
    checkpoint_path = resolve_path(checkpoint_path)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def main():
    global NUM_CHUNKS, NUM_SAMPLING_STEPS, HIGH_QUALITY_SAVE, TEXT_CFG_SCALE

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Infinite World - Action-Conditioned Video Generation')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (default: configs/infworld_config.yaml)')
    parser.add_argument('--prompts', type=str, default=None,
                        help='Path to prompts YAML file (default: prompts/demo.yaml)')
    parser.add_argument('--num_chunks', type=int, default=None,
                        help='Number of video chunks to generate (default: 2)')
    parser.add_argument('--low_memory', action='store_true',
                        help='Enable low memory mode (reduces quality/steps for lower VRAM usage)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: 42)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate per prompt (default: 5)')
    parser.add_argument('--vbench_index', type=int, default=None,
                        help='VBench sample index (legacy single-run mode); ignored when --num_samples > 1')
    parser.add_argument('--vbench_output_dir', type=str, default=None,
                        help='Directory to save VBench-named videos ({prompt}-{index}.mp4)')
    parser.add_argument('--type', type=str, default='scenery,indoor',
                        help='Comma-separated image_type values to include, e.g. "scenery,indoor" (default: scenery,indoor)')
    cmd_args = parser.parse_args()

    print("[InfWorld] Flags:")
    print(f"  --config           {cmd_args.config or '(default)'}")
    print(f"  --prompts          {cmd_args.prompts or '(default)'}")
    print(f"  --num_chunks       {cmd_args.num_chunks or '(default)'}")
    print(f"  --num_samples      {cmd_args.num_samples}")
    print(f"  --seed             {cmd_args.seed or '(default)'}")
    print(f"  --low_memory       {cmd_args.low_memory}")
    print(f"  --vbench_output_dir {cmd_args.vbench_output_dir or '(none)'}")
    print(f"  --type{cmd_args.type}")

    # Override seed if provided
    if cmd_args.seed is not None:
        setup_seed(cmd_args.seed + global_rank)

    # Update NUM_CHUNKS if provided via command line
    if cmd_args.num_chunks is not None:
        NUM_CHUNKS = cmd_args.num_chunks

    # Apply low memory optimizations
    if cmd_args.low_memory:
        NUM_SAMPLING_STEPS = 8  # Ultra-aggressive: 2 steps for maximum speed
        HIGH_QUALITY_SAVE = False  # Lower video quality
        TEXT_CFG_SCALE = 3  # Minimal guidance for extreme speed
        print("[InfWorld] LOW MEMORY MODE ENABLED (ULTRA-FAST):")
        print(f"  - Sampling steps: {NUM_SAMPLING_STEPS} (default: 30) - 15x FASTER!")
        print(f"  - High quality save: {HIGH_QUALITY_SAVE} (default: True)")
        print(f"  - CFG scale: {TEXT_CFG_SCALE} (default: 5.0) - MAXIMUM SPEED!")

    torch_gc()

    # Use command-line config if provided, otherwise use default
    config_path = cmd_args.config if cmd_args.config else CONFIG_PATH
    config_path = resolve_path(config_path)
    args = OmegaConf.load(config_path)
    checkpoint_path = resolve_path(args.get("checkpoint_path", "checkpoints/models/diffusion_pytorch_model.safetensors"))
    
    ckpt_step = extract_ckpt_step(checkpoint_path)
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_BASE, f"infworld-ckpt{ckpt_step}-step{NUM_SAMPLING_STEPS}-cfg{TEXT_CFG_SCALE}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[InfWorld] Loading checkpoint: {checkpoint_path}")
    print(f"[InfWorld] Config: {config_path}")
    print(f"[InfWorld] Output directory: {output_dir}")
    
    # Resolve relative paths in config for models that load from disk
    if hasattr(args, "vae_cfg") and "vae_pth" in args.vae_cfg:
        args.vae_cfg.vae_pth = resolve_path(args.vae_cfg.vae_pth)
    if hasattr(args, "text_encoder_cfg"):
        if "checkpoint_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.checkpoint_path = resolve_path(args.text_encoder_cfg.checkpoint_path)
        if "tokenizer_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.tokenizer_path = resolve_path(args.text_encoder_cfg.tokenizer_path)
    
    # Initialize models
    print("[InfWorld] Loading VAE...")
    vae = get_obj_from_str(args.vae_target)(**args.vae_cfg).to(local_rank)
    
    print("[InfWorld] Loading Text Encoder...")
    text_encoder = get_obj_from_str(args.text_encoder_target)(device=local_rank, **args.text_encoder_cfg)
    text_encoder.t5.model.to(local_rank)
    
    print("[InfWorld] Loading Scheduler...")
    scheduler = get_obj_from_str(args.scheduler_target)(**args.val_scheduler_cfg)
    scheduler.num_sampling_steps = NUM_SAMPLING_STEPS
    scheduler.shift = SHIFT
    
    print("[InfWorld] Loading DiT Model...")
    dtype = getattr(torch, args.amp_dtype)
    dit = get_obj_from_str(args.model_target)(
        out_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_context_parallel=enable_context_parallel,
        **args.model_cfg
    ).to(dtype)
    dit.eval()
    
    # Load DiT checkpoint (from config)
    state_dict = load_dit_state_dict(args.checkpoint_path)
    
    # Remove position embeddings (will be recomputed)
    state_dict.pop("pos_embed_temporal", None)
    state_dict.pop("pos_embed", None)
    
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    print(f"[InfWorld] Model loaded! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    dit.to(local_rank)
    
    # Load bucket config
    from infworld.configs import bucket_config as bucket_config_module
    bucket_config = getattr(bucket_config_module, BUCKET_CONFIG_NAME)
    
    # Load prompts - use command-line arg if provided, otherwise use default
    prompts_path = cmd_args.prompts if cmd_args.prompts else PROMPTS_YAML
    prompts_path = resolve_path(prompts_path)
    target_prompts = OmegaConf.load(prompts_path).prompts
    print(f"[InfWorld] Loaded {len(target_prompts)} prompts from: {prompts_path}")

    allowed_types = {t.strip() for t in cmd_args.type.split(",") if t.strip()} if cmd_args.type else None

    base_seed = cmd_args.seed if cmd_args.seed is not None else GLOBAL_SEED
    num_samples = cmd_args.num_samples

    # Pre-scan: report which prompts are already complete / skipped
    if cmd_args.vbench_output_dir:
        n_type_skip = n_already_done = n_partial = n_todo = 0
        for _idx, _entry in enumerate(target_prompts):
            _entry = list(_entry)
            _prompt = _entry[0]
            _image_type = _entry[3] if len(_entry) > 3 else ""
            if allowed_types is not None and _image_type not in allowed_types:
                n_type_skip += 1
                continue
            done = sum(
                1 for si in range(num_samples)
                if os.path.exists(os.path.join(cmd_args.vbench_output_dir, f"{_prompt}-{si}.mp4"))
            )
            if done == num_samples:
                print(f"[InfWorld] Exists task {_idx:4d} ({done}/{num_samples}): {_prompt[:70]}")
                n_already_done += 1
            elif done > 0:
                print(f"[InfWorld] Partial task {_idx:4d} ({done}/{num_samples}): {_prompt[:70]}")
                n_partial += 1
            else:
                n_todo += 1
        print(f"[InfWorld] Pre-scan: {len(target_prompts)} total | "
              f"{n_type_skip} type-filtered | {n_already_done} done | "
              f"{n_partial} partial | {n_todo} to generate")

    # Process each prompt
    _progress_done = 0
    _progress_total = n_todo + n_partial if cmd_args.vbench_output_dir else len(target_prompts)
    _progress_t0 = datetime.datetime.now()

    for task_idx, entry in enumerate(target_prompts):
        entry = list(entry)
        prompt, image_path, action_path = entry[0], entry[1], entry[2]
        image_type = entry[3] if len(entry) > 3 else ""

        if allowed_types is not None and image_type not in allowed_types:
            print(f"[InfWorld] Skipping task {task_idx} (type={image_type!r} not in {allowed_types}): {prompt[:50]}")
            continue
        if task_idx % dp_size != dp_rank:
            continue

        if not os.path.exists(image_path):
            print(f"[InfWorld] Skipping task {task_idx}: Image not found - {image_path}")
            continue

        if not os.path.exists(action_path):
            print(f"[InfWorld] Skipping task {task_idx}: Action not found - {action_path}")
            continue

        _progress_done += 1
        _elapsed = (datetime.datetime.now() - _progress_t0).total_seconds()
        if _progress_done > 1 and _elapsed > 0:
            _avg = _elapsed / (_progress_done - 1)
            _remaining = _progress_total - _progress_done + 1
            _eta_s = int(_avg * _remaining)
            _eta = f"{_eta_s // 3600}h {(_eta_s % 3600) // 60}m {_eta_s % 60}s"
        else:
            _eta = "?"
        print(f"[InfWorld] [{_progress_done}/{_progress_total}] Task {task_idx} | ETA {_eta} | {prompt[:50]}...")

        # Per-task output subdirectory
        task_subdir = os.path.join(output_dir, f"{task_idx:04d}_{prompt[:30].replace(' ', '_')}")
        if os.path.isdir(task_subdir):
            print(f"[InfWorld] Skipping task {task_idx}: subdir exists — {task_subdir}")
            continue
        os.makedirs(task_subdir, exist_ok=True)

        # Load condition image and encode once per prompt (shared across samples)
        cond_video = load_condition_image(image_path, bucket_config).to(local_rank)

        with torch.no_grad():
            cond_latent = vae.encode(cond_video)

        # Load action sequence
        move_indices, view_indices = load_action_sequence(action_path)

        # Latent size for generation
        latent_size = list(cond_latent.shape)
        latent_size[2] = 21  # Output frames per chunk
        latent_size = torch.Size(latent_size)

        for sample_idx in range(num_samples):
            # Determine VBench index: multi-sample uses sample_idx, legacy single-run uses vbench_index arg
            vbench_idx = sample_idx if num_samples > 1 else cmd_args.vbench_index

            # Skip if VBench output already exists
            if cmd_args.vbench_output_dir and vbench_idx is not None:
                vbench_path = os.path.join(cmd_args.vbench_output_dir, f"{prompt}-{vbench_idx}-{base_seed + vbench_idx}.mp4")
                if os.path.exists(vbench_path):
                    print(f"[InfWorld] Skipping task {task_idx} sample {vbench_idx}: already exists")
                    continue

            print(f"[InfWorld] Task {task_idx} sample {sample_idx + 1}/{num_samples}...")

            # Set seed per sample for reproducible diversity
            sample_seed = base_seed + sample_idx
            setup_seed(sample_seed + global_rank)

            # Reset video buffer for each sample
            video_buffer = cond_video.clone().cpu()

            # Generate video chunks
            chunk_paths = []
            for chunk_idx in range(NUM_CHUNKS):
                print(f"[InfWorld] Generating chunk {chunk_idx + 1}/{NUM_CHUNKS}")

                with torch.no_grad():
                    current_cond = video_buffer.to(local_rank)
                    current_latent = vae.encode(current_cond)

                # Get action slice for current chunk
                curr_start = video_buffer.shape[2] - 1
                curr_end = curr_start + args.validation_data.num_frames

                move = torch.tensor(move_indices[curr_start:curr_end], dtype=torch.long, device=local_rank)
                view = torch.tensor(view_indices[curr_start:curr_end], dtype=torch.long, device=local_rank)

                # Pad if needed
                num_frames = args.validation_data.num_frames
                if move.shape[0] < num_frames:
                    pad_len = num_frames - move.shape[0]
                    move = torch.cat([move, torch.zeros(pad_len, dtype=torch.long, device=local_rank)])
                    view = torch.cat([view, torch.zeros(pad_len, dtype=torch.long, device=local_rank)])

                additional_args = {
                    "image_cond": current_latent,
                    "move": move.unsqueeze(0),
                    "view": view.unsqueeze(0),
                }

                torch_gc()

                with torch.no_grad():
                    samples = scheduler.sample(
                        model=dit,
                        text_encoder=text_encoder,
                        null_embedder=dit.y_embedder,
                        z_size=latent_size,
                        prompts=[prompt],
                        guidance_scale=TEXT_CFG_SCALE,
                        negative_prompts=[NEGATIVE_PROMPT],
                        device=torch.device(local_rank),
                        additional_args=additional_args,
                    )

                    decoded_chunk = vae.decode(samples).cpu()

                    file_prefix = vbench_idx if vbench_idx is not None else task_idx
                    stem = prompt[:30].replace(' ', '_')
                    quality = 10 if HIGH_QUALITY_SAVE else 5

                    # Save individual chunk (only new frames)
                    individual_chunk_name = f"{file_prefix:04d}_{stem}_seed{sample_seed}_chunk{chunk_idx:03d}_individual"
                    individual_chunk_path = os.path.join(task_subdir, individual_chunk_name)
                    save_silent_video(decoded_chunk.to(local_rank), individual_chunk_path, fps=20, quality=quality)
                    print(f"[InfWorld] Saved individual chunk: {individual_chunk_path}.mp4")
                    chunk_paths.append(individual_chunk_path + ".mp4")

                    # Append to cumulative buffer
                    video_buffer = torch.cat([video_buffer, decoded_chunk[:, :, 1:]], dim=2)

                    print(f"[InfWorld] Chunk {chunk_idx + 1} done. Total frames: {video_buffer.shape[2]}")

                    # Save cumulative video (all frames up to this chunk)
                    cumulative_chunk_name = f"{file_prefix:04d}_{stem}_seed{sample_seed}_chunk{chunk_idx:03d}_cumulative"
                    cumulative_chunk_path = os.path.join(task_subdir, cumulative_chunk_name)
                    save_silent_video(video_buffer.to(local_rank), cumulative_chunk_path, fps=20, quality=quality)
                    print(f"[InfWorld] Saved cumulative: {cumulative_chunk_path}.mp4")
                    chunk_paths.append(cumulative_chunk_path + ".mp4")

                    torch_gc()

            # Save final video
            file_prefix = vbench_idx if vbench_idx is not None else task_idx
            stem = prompt[:30].replace(' ', '_')
            quality = 10 if HIGH_QUALITY_SAVE else 5
            save_path = os.path.join(task_subdir, f"{file_prefix:04d}_{stem}_seed{sample_seed}")
            save_silent_video(video_buffer.to(local_rank), save_path, fps=20, quality=quality)
            print(f"[InfWorld] Saved: {save_path}.mp4")

            # Delete intermediate chunk files now that final video is written
            for p in chunk_paths:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

            # VBench-compatible save: {prompt}-{index}.mp4 in flat output dir
            if vbench_idx is not None and cmd_args.vbench_output_dir:
                os.makedirs(cmd_args.vbench_output_dir, exist_ok=True)
                vbench_name = f"{prompt}-{vbench_idx}-{sample_seed}"
                vbench_path = os.path.join(cmd_args.vbench_output_dir, vbench_name)
                save_silent_video(video_buffer.to(local_rank), vbench_path, fps=20, quality=quality)
                print(f"[InfWorld] Saved VBench: {vbench_path}.mp4")

if __name__ == "__main__":
    main()
