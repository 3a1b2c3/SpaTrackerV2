# Disable Triton in xformers to avoid compatibility issues
import os
os.environ['XFORMERS_DISABLE_TRITON'] = '1'

# import pycolmap  # Not needed for RGB inference
from models.SpaTrackV2.models.predictor import Predictor
import yaml
import easydict
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import io
import moviepy.editor as mp
from models.SpaTrackV2.utils.visualizer import Visualizer
import tqdm
from models.SpaTrackV2.models.utils import get_points_on_a_grid
import glob
from rich import print
import argparse
import decord
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--data_type", type=str, default="RGBD")
    parser.add_argument("--data_dir", type=str, default="assets/example0")
    parser.add_argument("--video_name", type=str, default="snowboard")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--vo_points", type=int, default=756)
    parser.add_argument("--fps", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out_dir = args.data_dir + "/results"
    # fps
    fps = int(args.fps)
    mask_dir = args.data_dir + f"/{args.video_name}.png"

    print(f"\n{'='*60}")
    print(f"[bold cyan]SpatialTracker V2 Inference[/bold cyan]")
    print(f"{'='*60}")
    print(f"Video: [yellow]{args.video_name}[/yellow]")
    print(f"Data type: [yellow]{args.data_type}[/yellow]")
    print(f"Track mode: [yellow]{args.track_mode}[/yellow]")
    print(f"Grid size: [yellow]{args.grid_size}[/yellow]")
    print(f"Output: [yellow]{out_dir}[/yellow]")
    print(f"{'='*60}\n")

    print("[bold green]Step 1/6:[/bold green] Loading VGGT4Track model...")
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")
    print("[green]✓[/green] VGGT4Track model loaded\n")

    print("[bold green]Step 2/6:[/bold green] Loading video data...")
    if args.data_type == "RGBD":
        npz_dir = args.data_dir + f"/{args.video_name}.npz"
        data_npz_load = dict(np.load(npz_dir, allow_pickle=True))
        #TODO: tapip format
        video_tensor = data_npz_load["video"] * 255
        video_tensor = torch.from_numpy(video_tensor)
        video_tensor = video_tensor[::fps]
        depth_tensor = data_npz_load["depths"]
        depth_tensor = depth_tensor[::fps]
        intrs = data_npz_load["intrinsics"]
        intrs = intrs[::fps]
        extrs = np.linalg.inv(data_npz_load["extrinsics"])
        extrs = extrs[::fps]
        unc_metric = None
        print(f"[green]✓[/green] Loaded RGBD data ({len(video_tensor)} frames)\n")
    elif args.data_type == "RGB":
        vid_dir = os.path.join(args.data_dir, f"{args.video_name}.mp4")
        video_reader = decord.VideoReader(vid_dir)
        total_frames = len(video_reader)
        print(f"  Total frames in video: {total_frames}")
        video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)  # Convert to tensor and permute to (N, C, H, W)
        video_tensor = video_tensor[::fps].float()
        print(f"  Sampled frames (fps={fps}): {len(video_tensor)}")
        print(f"[green]✓[/green] Video loaded\n")

        # process the image tensor
        print("[bold green]Step 3/6:[/bold green] Predicting camera poses and depth...")
        print(f"  Video info: {len(video_tensor)} frames, {video_tensor.shape[2]}x{video_tensor.shape[3]} resolution")
        print(f"  Preprocessing frames for VGGT4Track...")
        video_tensor = preprocess_image(video_tensor)[None]
        video_shape = video_tensor.shape
        print(f"  Preprocessed shape: {video_shape[1]} frames, {video_shape[2]}x{video_shape[3]}x{video_shape[4]}")
        print(f"  Running VGGT4Track inference (bfloat16 precision)...")
        print(f"  [yellow]⏳ Computing: camera poses, intrinsics, depth maps, confidence metrics...[/yellow]")
        print(f"  [dim](This may take 1-3 minutes depending on video length and GPU)[/dim]")

        import time
        import threading

        # Progress indicator for long-running inference
        stop_progress = threading.Event()
        def show_progress():
            elapsed = 0
            while not stop_progress.is_set():
                time.sleep(5)  # Update every 5 seconds
                if not stop_progress.is_set():
                    elapsed += 5
                    print(f"  [dim]  ... still computing ({elapsed}s elapsed)[/dim]")

        progress_thread = threading.Thread(target=show_progress, daemon=True)

        start_time = time.time()
        progress_thread.start()

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Predict attributes including cameras, depth maps, and point maps.
                predictions = vggt4track_model(video_tensor.cuda()/255)
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]

        # Stop progress indicator
        stop_progress.set()
        progress_thread.join(timeout=1)

        inference_time = time.time() - start_time
        print(f"  [green]✓[/green] Inference completed in {inference_time:.1f} seconds ({inference_time/len(video_tensor):.2f}s per frame)")
        print(f"  Converting predictions to numpy arrays...")
        depth_tensor = depth_map.squeeze().cpu().numpy()
        extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
        extrs = extrinsic.squeeze().cpu().numpy()
        intrs = intrinsic.squeeze().cpu().numpy()
        video_tensor = video_tensor.squeeze()
        #NOTE: 20% of the depth is not reliable
        # threshold = depth_conf.squeeze()[0].view(-1).quantile(0.6).item()
        unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
        print(f"  Applying confidence threshold (>0.5) to depth predictions...")
        print(f"[green]✓[/green] Camera poses and depth predicted\n")

        data_npz_load = {}

    print("[bold green]Step 4/6:[/bold green] Loading tracking model...")
    if os.path.exists(mask_dir):
        mask_files = mask_dir
        mask = cv2.imread(mask_files)
        mask = cv2.resize(mask, (video_tensor.shape[3], video_tensor.shape[2]))
        mask = mask.sum(axis=-1)>0
    else:
        mask = np.ones_like(video_tensor[0,0].numpy())>0
        
    # get all data pieces
    viz = True
    os.makedirs(out_dir, exist_ok=True)
        
    # with open(cfg_dir, "r") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg = easydict.EasyDict(cfg)
    # cfg.out_dir = out_dir
    # cfg.model.track_num = args.vo_points
    # print(f"Downloading model from HuggingFace: {cfg.ckpts}")
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")

    # config the model; the track_num is the number of points in the grid
    model.spatrack.track_num = args.vo_points
    
    model.eval()
    model.to("cuda")
    print(f"[green]✓[/green] Tracking model loaded\\n")

    viser = Visualizer(save_dir=out_dir, grayscale=True,
                     fps=10, pad_value=0, tracks_leave_trace=5)
    
    grid_size = args.grid_size

    # get frame H W
    if video_tensor is  None:
        cap = cv2.VideoCapture(video_path)
        frame_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    
    # Sample mask values at grid points and filter out points where mask=0
    if os.path.exists(mask_dir):
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int[...,1], grid_pts_int[...,0]]
        grid_pts = grid_pts[:, mask_values]
    
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

    print(f"[bold green]Step 5/6:[/bold green] Running tracking inference...")
    print(f"  Tracking {len(query_xyt)} points across {len(video_tensor)} frames")

    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2)

        print(f"[green]✓[/green] Tracking inference completed\\n")

        print("[bold green]Step 6/6:[/bold green] Processing and saving results...")

        # resize the results to avoid too large I/O Burden
        # depth and image, the maximum side is 336
        max_size = 336
        h, w = video.shape[2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            video = T.Resize((new_h, new_w))(video)
            video_tensor = T.Resize((new_h, new_w))(video_tensor)
            point_map = T.Resize((new_h, new_w))(point_map)
            conf_depth = T.Resize((new_h, new_w))(conf_depth)
            track2d_pred[...,:2] = track2d_pred[...,:2] * scale
            intrs[:,:2,:] = intrs[:,:2,:] * scale
            if depth_tensor is not None:
                if isinstance(depth_tensor, torch.Tensor):
                    depth_tensor = T.Resize((new_h, new_w))(depth_tensor)
                else:
                    depth_tensor = T.Resize((new_h, new_w))(torch.from_numpy(depth_tensor))

        if viz:
            viser.visualize(video=video[None],
                                tracks=track2d_pred[None][...,:2],
                                visibility=vis_pred[None],filename=args.video_name)

        # save as the tapip3d format   
        data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_npz_load["intrinsics"] = intrs.cpu().numpy()
        depth_save = point_map[:,2,...]
        depth_save[conf_depth<0.5] = 0
        data_npz_load["depths"] = depth_save.cpu().numpy()
        data_npz_load["video"] = (video_tensor).cpu().numpy()/255
        data_npz_load["visibs"] = vis_pred.cpu().numpy()
        data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
        result_filename = f'{args.video_name}_result.npz'
        np.savez(os.path.join(out_dir, result_filename), **data_npz_load)

        print(f"[green]✓[/green] Results saved to [yellow]{out_dir}/{result_filename}[/yellow]")
        print(f"\n{'='*60}")
        print(f"[bold green]Inference Complete![/bold green]")
        print(f"{'='*60}")
        print(f"To visualize results with tapip3d, run:")
        print(f"[bold yellow]python tapip3d_viz.py {out_dir}/{result_filename}[/bold yellow]\n")
