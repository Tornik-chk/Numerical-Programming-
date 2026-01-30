"""
Drone Show Project — Sub-problem 3 (Dynamic Tracking and Shape Preservation)

MATHEMATICAL FORMULATION:
========================
This module solves a time-varying shape-tracking IVP for dynamic object following.

State Variables:
  - x_i(t) ∈ ℝ² : position of drone i at time t
  - v_i(t) ∈ ℝ² : velocity of drone i at time t

System of ODEs (time-varying target extracted from video):
  dx_i/dt = sat(v_i, v_max)
  dv_i/dt = (1/m)[k_p(T_i(t) - x_i) + ∑_{j≠i} F_rep(x_i, x_j) - k_d·v_i]

Time-Varying Target (from video contours):
  T_i(t) = contour points from video frame at time t
  
  Each frame:
    1. Extract largest moving object contour from video
    2. Sample N points uniformly along contour perimeter
    3. Map pixel coordinates to simulation space
    4. Assign targets to drones using Hungarian/greedy algorithm

Key Differences from Sub-problems 1-2:
  - Targets change EVERY frame (not just linear interpolation)
  - Target shape morphs dynamically based on object appearance
  - Assignment problem solved per-frame to prevent "cloud effect"

Initial Conditions (t=0):
  - x_i(0) = final positions from Sub-problem 2 ("Happy New Year!")
  - v_i(0) = final velocities from Sub-problem 2

Object Detection Strategy:
  - Binary thresholding (adjustable for bright/dark objects)
  - Contour detection (largest by area or closest to previous)
  - Arc-length parameterization for uniform point sampling
  - Temporal smoothing to reduce jitter

NUMERICAL METHOD:
=================
Same RK4 implementation, but with:
  - Frame-interpolated targets T(t) between video frames
  - Pre-computed assignment to avoid O(N²) cost in RK4 stages

PHYSICAL INTERPRETATION:
========================
The swarm forms the dynamic shape of a moving object:
  - Higher k_p needed for fast shape changes
  - Temporal smoothing trades responsiveness for stability
  - Assignment algorithm prevents drones from crossing unnecessarily

REQUIREMENTS:
=============
pip install numpy opencv-python matplotlib scipy

INPUT:
======
- trajectories_subproblem2.npy (output from Sub-problem 2)
- Video file with moving object (bright on dark or vice versa)
- Same N_DRONES parameter

OUTPUT:
=======
- trajectories_subproblem3.npy (trajectory array)
- Animated visualization showing swarm tracking object shape

TUNING GUIDE:
=============
For bright object on dark background:
  - invert=False, thresh=150-230

For dark object on bright background:
  - invert=True, thresh=120-200

For noisy video:
  - Increase blur_ksize (7, 9, 11)
  - Increase smooth_win (5, 7, 9)

For fast-moving object:
  - Increase kp (attract faster)
  - Increase vmax (allow higher speeds)

AI USAGE DISCLOSURE:
====================
Claude AI (Anthropic) was used for:
- Code structure and optimization
- Documentation and mathematical formulation
- Contour processing and arc-length sampling algorithms
- Hungarian algorithm integration for optimal assignment
- Video processing pipeline design
"""

from __future__ import annotations
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist


# ============================
# CONFIGURATION
# ============================
DEBUG_PLOTS = False


# ========================================
# DYNAMICS (reused from Sub-problems 1-2)
# ========================================

def v_saturate(v: np.ndarray, vmax: float) -> np.ndarray:
    """Saturate velocity to max speed."""
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    scale = np.ones_like(speed)
    mask = speed > 1e-12
    scale[mask] = np.minimum(1.0, vmax / speed[mask])
    return v * scale


def repulsion_forces(
    x: np.ndarray,
    krep: float,
    rsafe: float,
    eps: float = 1e-6,
    use_spatial_index: bool = True,
) -> np.ndarray:
    """
    Pairwise repulsion forces for collision avoidance.
    Same as Sub-problems 1-2.
    """
    n = x.shape[0]
    f = np.zeros_like(x)
    
    if use_spatial_index and n > 50:
        tree = cKDTree(x)
        for i in range(n):
            indices = tree.query_ball_point(x[i], rsafe)
            for j in indices:
                if i != j:
                    dx = x[i] - x[j]
                    dist = np.linalg.norm(dx) + eps
                    if dist < rsafe:
                        f[i] += krep * dx / (dist ** 3)
    else:
        # Naive O(N²) for small N
        dx = x[:, None, :] - x[None, :, :]
        dist2 = np.sum(dx * dx, axis=2) + eps
        dist = np.sqrt(dist2)
        
        np.fill_diagonal(dist, np.inf)
        
        mask = dist < rsafe
        inv_dist3 = np.zeros_like(dist)
        inv_dist3[mask] = 1.0 / (dist[mask] ** 3)
        
        f = krep * np.sum(dx * inv_dist3[:, :, None], axis=1)
    
    return f


# ========================================
# VIDEO PROCESSING & CONTOUR EXTRACTION
# ========================================

def sample_contour_points(contour: np.ndarray, n: int) -> np.ndarray:
    """
    Sample n points uniformly along contour perimeter using arc-length parameterization.
    
    This ensures uniform spacing regardless of contour point density.
    
    Args:
        contour: (M, 1, 2) OpenCV contour array
        n: Number of points to sample
    
    Returns:
        (n, 2) uniformly sampled points along contour
    """
    pts = contour[:, 0, :].astype(float)
    if len(pts) == 0:
        return np.zeros((n, 2), dtype=float)

    # Compute arc-length parameterization
    seg = np.diff(pts, axis=0, append=pts[:1])  # Segments (close the loop)
    d = np.linalg.norm(seg, axis=1)              # Segment lengths
    s = np.concatenate([[0.0], np.cumsum(d)])    # Cumulative arc-length
    L = float(s[-1])                              # Total perimeter
    
    if L < 1e-9:
        # Degenerate contour (single point)
        return np.repeat(pts[:1], n, axis=0)

    # Normalize to [0, 1]
    s = s / L
    
    # Sample uniformly in arc-length space
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    out = np.zeros((n, 2), dtype=float)

    # Find points at each arc-length position
    j = 0
    for i, ui in enumerate(u):
        while j + 1 < len(s) and s[j + 1] <= ui:
            j += 1
        # Linear interpolation between pts[j] and pts[j+1]
        if j + 1 < len(pts):
            alpha = (ui - s[j]) / (s[j + 1] - s[j] + 1e-12)
            out[i] = (1 - alpha) * pts[j] + alpha * pts[j + 1]
        else:
            out[i] = pts[j]
    
    return out


def contour_centroid(c: np.ndarray) -> np.ndarray | None:
    """
    Compute centroid of contour using image moments.
    
    Args:
        c: OpenCV contour
    
    Returns:
        (2,) centroid coordinates, or None if degenerate
    """
    M = cv2.moments(c)
    if abs(M["m00"]) < 1e-9:
        return None
    return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=float)


def select_contour(
    contours: list,
    prev_centroid: np.ndarray | None = None,
    min_area: int = 200
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Select the best contour from a list.
    
    Strategy:
      - Filter by minimum area
      - If no previous: pick most compact (area/bbox_area ratio)
      - If previous exists: pick closest to previous centroid (temporal coherence)
    
    Args:
        contours: List of OpenCV contours
        prev_centroid: Previous frame's object centroid (for tracking)
        min_area: Minimum contour area to consider
    
    Returns:
        (selected_contour, centroid) or (None, prev_centroid) if no valid contour
    """
    candidates = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= min_area]
    
    if not candidates:
        return None, prev_centroid

    if prev_centroid is None:
        # First frame: pick most compact blob (avoids selecting noise)
        best = None
        best_score = -1.0
        best_cent = None
        
        for c, area in candidates:
            x, y, w, h = cv2.boundingRect(c)
            bbox_area = w * h + 1e-9
            compactness = area / bbox_area
            
            if compactness > best_score:
                best_score = compactness
                best = c
                best_cent = contour_centroid(c)
        
        return best, best_cent
    
    # Subsequent frames: pick closest to previous centroid (temporal tracking)
    best = None
    best_dist = 1e18
    best_cent = prev_centroid
    
    for c, area in candidates:
        cent = contour_centroid(c)
        if cent is None:
            continue
        
        dist = float(np.sum((cent - prev_centroid) ** 2))
        if dist < best_dist:
            best_dist = dist
            best = c
            best_cent = cent
    
    return best, best_cent


def order_points_by_angle(pts: np.ndarray) -> np.ndarray:
    """
    Order points by polar angle around centroid.
    This creates a consistent winding order for contour points.
    
    Args:
        pts: (N, 2) points
    
    Returns:
        (N, 2) points sorted by angle
    """
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def track_shape_video(
    video_path: str,
    n_points: int,
    thresh: int = 200,
    invert: bool = False,
    blur_ksize: int = 7,
    max_frames: int | None = None,
    min_area: int = 200,
) -> tuple[np.ndarray, float, tuple[int, int], list[np.ndarray]]:
    """
    Extract dynamic shape from video by tracking largest moving object contour.
    
    Pipeline:
      1. Read video frame-by-frame
      2. Convert to grayscale and blur
      3. Binary threshold (bright or dark object)
      4. Find contours
      5. Select best contour (largest/closest to previous)
      6. Sample n_points uniformly along contour
      7. Order points by angle for consistency
    
    Args:
        video_path: Path to video file
        n_points: Number of points to sample per frame
        thresh: Binary threshold value (0-255)
        invert: If True, invert threshold (dark object on bright background)
        blur_ksize: Gaussian blur kernel size (must be odd)
        max_frames: Maximum frames to process (None = all)
        min_area: Minimum contour area to consider
    
    Returns:
        shapes: (F, n_points, 2) pixel coordinates of contour points per frame
        fps: Video frames per second
        frame_size: (width, height) of video
        frames_rgb: List of RGB frames for visualization
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not (fps > 0):
        fps = 30.0
        print(f"Warning: Invalid FPS from video, using default {fps}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    frames_rgb = []
    shapes = []
    last_good = None
    prev_centroid = None

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        frame_count += 1
        if max_frames is not None and frame_count > max_frames:
            break

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        # Binary threshold
        if invert:
            _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        else:
            _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        bw = cv2.medianBlur(bw, 3)

        # Find contours
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Select best contour
        selected, prev_centroid = select_contour(contours, prev_centroid, min_area)

        if selected is None:
            # No valid contour: use last good shape
            shape = last_good if last_good is not None else np.zeros((n_points, 2), dtype=float)
        else:
            # Sample and order points
            shape = sample_contour_points(selected, n_points)
            shape = order_points_by_angle(shape)
            last_good = shape

        shapes.append(shape)
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    
    print(f"Processed {len(shapes)} frames from video")
    return np.array(shapes, dtype=float), fps, (W, H), frames_rgb


# ========================================
# COORDINATE TRANSFORMATION
# ========================================

def px_points_to_sim(
    T_frames_px: np.ndarray,
    frame_size: tuple[int, int],
    sim_scale: float = 10.0
) -> np.ndarray:
    """
    Convert pixel coordinates to simulation coordinates.
    
    Pixel space: origin at top-left, x right, y down
    Sim space:   origin at center, x right, y up
    
    Args:
        T_frames_px: (F, N, 2) pixel coordinates
        frame_size: (width, height) of video
        sim_scale: Scale factor for simulation space
    
    Returns:
        (F, N, 2) simulation coordinates
    """
    W, H = frame_size
    out = T_frames_px.copy()
    
    # Center and normalize x
    out[..., 0] = (out[..., 0] - W / 2) / (W / 2) * (sim_scale / 2)
    
    # Center, normalize, and flip y (pixel y-down to math y-up)
    out[..., 1] = -(out[..., 1] - H / 2) / (H / 2) * (sim_scale / 2)
    
    return out


def smooth_targets(T: np.ndarray, win: int = 5) -> np.ndarray:
    """
    Temporal smoothing of target trajectories to reduce jitter.
    
    Uses simple moving average over time frames.
    
    Args:
        T: (F, N, 2) target positions over frames
        win: Smoothing window size (must be odd)
    
    Returns:
        (F, N, 2) smoothed targets
    """
    if win <= 1:
        return T
    
    if win % 2 == 0:
        win += 1
    
    pad = win // 2
    
    # Pad at boundaries with edge values
    Tp = np.pad(T, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    
    out = np.zeros_like(T)
    for k in range(len(T)):
        out[k] = Tp[k:k + win].mean(axis=0)
    
    return out


# ========================================
# TARGET ASSIGNMENT (prevent cloud effect)
# ========================================

def greedy_match_targets(x: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Assign targets to drones using Hungarian algorithm (optimal) or greedy fallback.
    
    The assignment problem:
      Minimize total squared distance between current positions and targets.
    
    Hungarian algorithm: O(N³) but optimal
    Greedy fallback: O(N²) but suboptimal
    
    Args:
        x: (N, 2) current drone positions
        targets: (N, 2) target positions
    
    Returns:
        (N, 2) reordered targets, one per drone
    """
    try:
        from scipy.optimize import linear_sum_assignment
        
        # Cost matrix: squared distances
        cost = np.sum((x[:, None, :] - targets[None, :, :]) ** 2, axis=2)
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost)
        
        return targets[col_ind]
    
    except ImportError:
        # Scipy not available: use greedy algorithm
        print("Warning: scipy not available, using greedy assignment (suboptimal)")
        
        remaining = targets.copy()
        assigned = np.zeros_like(targets)
        
        for i in range(x.shape[0]):
            d2 = np.sum((remaining - x[i]) ** 2, axis=1)
            j = int(np.argmin(d2))
            assigned[i] = remaining[j]
            remaining = np.delete(remaining, j, axis=0)
        
        return assigned


def assign_targets_per_frame(
    T_frames: np.ndarray,
    x_init: np.ndarray
) -> np.ndarray:
    """
    Pre-compute optimal target assignment for all frames.
    
    This prevents the "cloud effect" where drones swap targets chaotically.
    By computing assignment once based on initial positions, we ensure
    smooth transitions.
    
    Args:
        T_frames: (F, N, 2) target positions per frame
        x_init: (N, 2) initial drone positions
    
    Returns:
        (F, N, 2) assigned targets per frame
    """
    assigned = np.zeros_like(T_frames)
    x_ref = x_init.copy()
    
    for k in range(len(T_frames)):
        assigned[k] = greedy_match_targets(x_ref, T_frames[k])
        # Update reference to current assignment (temporal coherence)
        x_ref = assigned[k]
    
    return assigned


def interpolate_targets(T_frames: np.ndarray, fps: float, t: float) -> np.ndarray:
    """
    Linear interpolation of targets between video frames.
    
    This provides smooth target motion at simulation timestep resolution.
    
    Args:
        T_frames: (F, N, 2) target positions per frame
        fps: Video frames per second
        t: Current simulation time
    
    Returns:
        (N, 2) interpolated target positions at time t
    """
    k = t * fps  # Frame index (continuous)
    k0 = int(np.floor(k))
    k0 = max(0, min(k0, len(T_frames) - 1))
    k1 = min(k0 + 1, len(T_frames) - 1)
    
    alpha = float(k - k0)  # Interpolation weight
    
    return (1 - alpha) * T_frames[k0] + alpha * T_frames[k1]


# ========================================
# DYNAMICS WITH VIDEO TARGETS
# ========================================

def rhs_shape(
    t: float,
    x: np.ndarray,
    v: np.ndarray,
    T_assigned: np.ndarray,
    fps: float,
    params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Right-hand side for shape-tracking dynamics.
    
    Same physics as Sub-problems 1-2, but targets come from video.
    
    Args:
        t: Current time
        x: (N, 2) positions
        v: (N, 2) velocities
        T_assigned: (F, N, 2) pre-assigned targets per frame
        fps: Video framerate
        params: Physics parameters
    
    Returns:
        xdot: (N, 2) velocity derivatives
        vdot: (N, 2) acceleration derivatives
    """
    m = params["m"]
    kp = params["kp"]
    kd = params["kd"]
    vmax = params["vmax"]
    krep = params["krep"]
    rsafe = params["rsafe"]

    # Get interpolated target for current time
    T = interpolate_targets(T_assigned, fps, t)

    xdot = v_saturate(v, vmax)
    frep = repulsion_forces(x, krep=krep, rsafe=rsafe)
    vdot = (1.0 / m) * (kp * (T - x) + frep - kd * v)
    
    return xdot, vdot


def rk4_step_shape(
    t: float,
    x: np.ndarray,
    v: np.ndarray,
    dt: float,
    T_assigned: np.ndarray,
    fps: float,
    params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    RK4 step for shape-tracking dynamics.
    """
    k1x, k1v = rhs_shape(t, x, v, T_assigned, fps, params)
    
    k2x, k2v = rhs_shape(
        t + 0.5 * dt,
        x + 0.5 * dt * k1x,
        v + 0.5 * dt * k1v,
        T_assigned, fps, params
    )
    
    k3x, k3v = rhs_shape(
        t + 0.5 * dt,
        x + 0.5 * dt * k2x,
        v + 0.5 * dt * k2v,
        T_assigned, fps, params
    )
    
    k4x, k4v = rhs_shape(
        t + dt,
        x + dt * k3x,
        v + dt * k3v,
        T_assigned, fps, params
    )

    x_next = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_next = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    
    return x_next, v_next


def simulate_shape_tracking(
    x0: np.ndarray,
    v0: np.ndarray,
    T_frames_sim: np.ndarray,
    fps: float,
    params: dict,
    dt: float = 0.01,
    settling_time: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate drone swarm tracking dynamic video object.
    
    Two phases:
    1. SETTLING: Drones align to first frame target (video frozen)
    2. TRACKING: Drones follow video dynamics
    
    Pre-computes target assignment to avoid O(N³) cost in RK4 inner loop.
    
    Args:
        x0: (N, 2) initial positions
        v0: (N, 2) initial velocities
        T_frames_sim: (F, N, 2) target shapes in simulation coordinates
        fps: Video framerate
        params: Physics parameters
        dt: Simulation timestep
        settling_time: Duration (seconds) to let drones settle on first frame
    
    Returns:
        traj: (T, N, 2) drone trajectories
        tgrid: (T,) time values
    """
    print("Pre-computing target assignments (this may take a moment)...")
    
    # Pre-assign targets ONCE based on initial positions
    T_assigned = assign_targets_per_frame(T_frames_sim, x0)
    
    print("Assignment complete, starting simulation...")
    
    # ---- PHASE 1: SETTLING (drones align to first frame) ----
    settling_steps = int(math.ceil(settling_time / dt))
    settling_grid = np.arange(settling_steps + 1) * dt
    
    settling_traj = np.zeros((settling_steps + 1, x0.shape[0], 2), dtype=float)
    x_settle = x0.copy()
    v_settle = v0.copy()
    settling_traj[0] = x_settle
    
    # Create frozen target (only first frame)
    T_first_frame = T_assigned[0:1]  # (1, N, 2)
    
    print(f"PHASE 1: Settling drones on first frame target ({settling_time:.1f}s)...")
    
    for k in range(settling_steps):
        # Interpolate on frozen first frame
        x_settle, v_settle = rk4_step_shape(
            settling_grid[k], x_settle, v_settle, dt, T_first_frame, fps, params
        )
        settling_traj[k + 1] = x_settle
    
    # ---- PHASE 2: TRACKING (video plays, drones follow) ----
    t_end = (len(T_frames_sim) - 1) / fps
    tracking_steps = int(math.ceil(t_end / dt))
    tracking_grid = np.arange(tracking_steps + 1) * dt
    
    tracking_traj = np.zeros((tracking_steps + 1, x_settle.shape[0], 2), dtype=float)
    x = x_settle.copy()
    v = v_settle.copy()
    tracking_traj[0] = x
    
    print(f"PHASE 2: Tracking video dynamics ({t_end:.1f}s)...")
    
    for k in range(tracking_steps):
        x, v = rk4_step_shape(tracking_grid[k], x, v, dt, T_assigned, fps, params)
        tracking_traj[k + 1] = x
    
    # Combine both phases
    # Shift tracking time to start after settling
    combined_traj = np.vstack([settling_traj, tracking_traj[1:]])
    combined_tgrid = np.concatenate([settling_grid, settling_time + tracking_grid[1:]])
    
    print(f"Simulation complete: {len(combined_tgrid)} total timesteps")
    
    return combined_traj, combined_tgrid


# ========================================
# VALIDATION
# ========================================

def validate_shape_tracking(
    traj: np.ndarray,
    T_frames: np.ndarray,
    fps: float,
    rsafe: float,
    dt: float,
    verbose: bool = True
) -> dict:
    """
    Validate shape-tracking simulation results.
    
    Metrics:
      1. Mean tracking error over time
      2. Collision safety
      3. Responsiveness (lag between target and swarm)
    
    Args:
        traj: (T, N, 2) trajectories
        T_frames: (F, N, 2) target shapes
        fps: Video framerate
        rsafe: Safety radius
        dt: Simulation timestep
        verbose: Print results
    
    Returns:
        Dictionary with validation metrics
    """
    # Sample tracking error at video frame times
    errors = []
    for frame_idx in range(len(T_frames)):
        t = frame_idx / fps
        traj_idx = int(t / dt)
        if traj_idx >= len(traj):
            break
        
        err = np.mean(np.linalg.norm(traj[traj_idx] - T_frames[frame_idx], axis=1))
        errors.append(err)
    
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    
    # Collision check
    min_dist_overall = float('inf')
    for t in range(traj.shape[0]):
        min_dist_t = float(np.min(pdist(traj[t])))
        min_dist_overall = min(min_dist_overall, min_dist_t)
    
    collision_occurred = min_dist_overall < rsafe
    
    results = {
        "mean_tracking_error": mean_error,
        "max_tracking_error": max_error,
        "min_distance_overall": min_dist_overall,
        "collision_occurred": collision_occurred,
    }
    
    if verbose:
        print("\n" + "="*50)
        print("VALIDATION RESULTS - SUB-PROBLEM 3")
        print("="*50)
        print(f"Tracking Performance:")
        print(f"  Mean error:     {mean_error:.4f}")
        print(f"  Max error:      {max_error:.4f}")
        print(f"\nCollision Safety:")
        print(f"  Min distance:   {min_dist_overall:.4f}")
        print(f"  Safety radius:  {rsafe:.4f}")
        print(f"  Status:         {'COLLISION' if collision_occurred else 'SAFE'}")
        print("="*50 + "\n")
    
    return results


def plot_tracking_metrics(traj: np.ndarray, T_frames: np.ndarray, fps: float, dt: float):
    """
    Plot tracking error over time.
    """
    errors = []
    times = []
    
    for frame_idx in range(len(T_frames)):
        t = frame_idx / fps
        traj_idx = int(t / dt)
        if traj_idx >= len(traj):
            break
        
        err = np.mean(np.linalg.norm(traj[traj_idx] - T_frames[frame_idx], axis=1))
        errors.append(err)
        times.append(t)
    
    plt.figure(figsize=(10, 5))
    plt.plot(times, errors, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Mean tracking error')
    plt.title('Shape Tracking Performance Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/sub3/tracking_metrics_subproblem3.png', dpi=150)
    print("Saved: tracking_metrics_subproblem3.png")
    plt.show()


# ========================================
# VISUALIZATION
# ========================================

def animate_swarm(
    traj: np.ndarray,
    all_x0: np.ndarray | None = None,
    drone_indices: np.ndarray | None = None,
    T_frames: np.ndarray | None = None,
    frames_rgb: list | None = None,
    fps: float | None = None,
    dt: float = 0.01,
    interval_ms: int = 25,
    settling_time: float = 3.0,
    save_mp4: bool = False,
    title: str = "Sub-problem 3: Dynamic Shape Tracking"
):
    """
    Animated visualization of swarm tracking dynamic shape.
    
    Two phases:
    1. SETTLING: Video and target frozen on first frame
    2. TRACKING: Video plays, drones follow shape
    
    If all_x0 and drone_indices provided, shows all drones with tracked ones animated
    and untracked ones fading away.
    
    Optionally shows target shape overlay for comparison.
    """
    Tsteps, N, _ = traj.shape

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect("equal", adjustable="box")
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Compute bounds
    xy = traj.reshape(-1, 2)
    if all_x0 is not None:
        xy = np.vstack([xy, all_x0])
    if T_frames is not None:
        xy = np.vstack([xy, T_frames.reshape(-1, 2)])
    
    pad = 0.5
    xmin, ymin = xy.min(axis=0) - pad
    xmax, ymax = xy.max(axis=0) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, color='white', fontsize=14)
    ax.tick_params(colors='white')

    # Background image (if video frames provided)
    img_obj = None
    if frames_rgb is not None and len(frames_rgb) > 0:
        sim_scale = 10.0
        extent = [-(sim_scale/2), (sim_scale/2), -(sim_scale/2), (sim_scale/2)]
        img_obj = ax.imshow(frames_rgb[0], extent=extent, origin='upper', alpha=0.3, cmap='gray', zorder=0)

    # Target shape (if provided)
    target_scat = None
    if T_frames is not None and fps is not None:
        target_scat = ax.scatter(
            [], [], s=8, c='cyan', alpha=0.3, marker='x',
        label='Target shape'
    ) 
    
    # Drones being tracked
    scat = ax.scatter(
        traj[0, :, 0], traj[0, :, 1],
        s=25, c='yellow', edgecolors='white',
        linewidths=0.5, alpha=0.9,
        label='Tracked drones'
    )
    
    # Untracked drones (fading away)
    fading_scat = None
    if all_x0 is not None and drone_indices is not None:
        untracked_mask = np.ones(len(all_x0), dtype=bool)
        untracked_mask[drone_indices] = False
        untracked_pos = all_x0[untracked_mask]
        fading_scat = ax.scatter(
            untracked_pos[:, 0], untracked_pos[:, 1],
            s=25, c='gray', edgecolors='white',
            linewidths=0.5, alpha=0.3,
            label='Fading drones'
        )

    # Motion trails
    Ktrail = 10
    trail_lines = [
        ax.plot([], [], linewidth=1, color='yellow', alpha=0.1)[0]
        for _ in range(Ktrail)
    ]

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        color='white', fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

    def update(frame: int):
        pts = traj[frame]
        scat.set_offsets(pts)

        # Calculate time and determine phase
        t = frame * dt
        
        # PHASE 1: SETTLING (first frame frozen)
        if t < settling_time:
            phase_text = "PHASE 1: SETTLING (Video Frozen)"
            frame_idx = 0  # Always show first frame
        # PHASE 2: TRACKING (video plays)
        else:
            phase_text = "PHASE 2: TRACKING (Video Playing)"
            # Calculate video frame based on elapsed time after settling
            elapsed = t - settling_time
            frame_idx = int(elapsed * fps) if fps is not None else 0
            if frame_idx >= len(T_frames):
                frame_idx = len(T_frames) - 1

        # Update target shape (frozen or dynamic based on phase)
        if target_scat is not None and T_frames is not None:
            if frame_idx < len(T_frames):
                target_scat.set_offsets(T_frames[frame_idx])

        # Update video background (frozen or dynamic based on phase)
        if img_obj is not None and frames_rgb is not None:
            if frame_idx < len(frames_rgb):
                img_obj.set_array(frames_rgb[frame_idx])

        # Update trails
        for i, ln in enumerate(trail_lines):
            f0 = max(0, frame - (i + 1) * 2)
            f1 = frame
            if f1 > f0:
                cseg = traj[f0:f1 + 1].mean(axis=1)
                ln.set_data(cseg[:, 0], cseg[:, 1])

        # Hide fading drones after settling phase ends
        if fading_scat is not None:
            if t >= settling_time:
                fading_scat.set_visible(False)

        time_text.set_text(f'{phase_text}\nFrame: {frame + 1}/{Tsteps}  Time: {frame * dt:.2f}s')
        
        ret = [scat]
        if img_obj is not None:
            ret.append(img_obj)
        if target_scat is not None:
            ret.append(target_scat)
        if fading_scat is not None:
            ret.append(fading_scat)
        ret.extend(trail_lines)
        ret.append(time_text)
        return tuple(ret)

    anim = FuncAnimation(fig, update, frames=Tsteps, interval=interval_ms, blit=False)

    if save_mp4:
        print("Saving animation...")
        anim.save("subproblem3_shape_tracking.mp4", dpi=150, writer='ffmpeg')
        print("Saved: subproblem3_shape_tracking.mp4")

    plt.show()
    return anim
        

# ==============================
# MAIN EXECUTION
# ==============================

# Input paths
VIDEO_PATH = os.path.join("data", "Ball2.mp4")
TRAJ2_PATH = "outputs/sub2/trajectories_subproblem2.npy"

# Simulation settings
dt_sim = 0.01
sim_scale = 10.0

# Video processing settings (TUNE THESE FOR YOUR VIDEO)
# For bright object on dark background:
#   thresh=150-230, invert=False
# For dark object on bright background:
#   thresh=120-200, invert=True
thresh = 230
invert = False
blur_ksize = 7
min_area = 200
smooth_win = 20

# Physics parameters (higher kp for faster tracking)
params = dict(
    m=1.0,
    kp=70.0,     # Higher than Sub-1/2 for fast shape changes
    kd=9.0,      # Moderate damping
    vmax=3.0,    # Allow higher speeds for tracking
    krep=1,    # Stronger repulsion (denser formations)
    rsafe=0.1,
) 

# ============================================
# DRONE TRACKING CONFIGURATION
# ============================================
# Set how many drones you want to track the object (rest will fade away)
# If None, all drones from Sub-problem 2 will be used
N_DRONES_TRACK = 50  # <-- CHANGE THIS to control number of tracking drones

print("="*60)
print("DRONE SHOW - SUB-PROBLEM 3: DYNAMIC SHAPE TRACKING")
print("="*60)
print(f"Configuration:")
print(f"  Video: {VIDEO_PATH}")
print(f"  Sub-2 trajectories: {TRAJ2_PATH}")
print(f"  Threshold: {thresh}, Invert: {invert}")
print(f"  Physics params: {params}")
print("="*60 + "\n")

# ============================================
# LOAD FINAL STATE FROM SUB-PROBLEM 2
# ============================================
print("Loading Sub-problem 2 results...")

if not os.path.exists(TRAJ2_PATH):
    raise FileNotFoundError(
        f"Cannot find {TRAJ2_PATH}. "
        f"Please run Sub-problem 2 first!"
    )

traj2 = np.load(TRAJ2_PATH)
all_x0 = traj2[-1].copy()
all_v0 = (traj2[-1] - traj2[-2]) / dt_sim

N_DRONES_TOTAL = all_x0.shape[0]

print(f"  OK Loaded final state: {N_DRONES_TOTAL} drones from Sub-problem 2")

# ============================================
# RANDOMLY SAMPLE DRONES TO TRACK
# ============================================
if N_DRONES_TRACK is None or N_DRONES_TRACK >= N_DRONES_TOTAL:
    N_DRONES_TRACK = N_DRONES_TOTAL
    x0 = all_x0
    v0 = all_v0
    drone_indices = np.arange(N_DRONES_TOTAL)
else:
    # Randomly select which drones to track
    rng = np.random.default_rng(seed=42)
    drone_indices = rng.choice(N_DRONES_TOTAL, size=N_DRONES_TRACK, replace=False)
    drone_indices = np.sort(drone_indices)
    
    x0 = all_x0[drone_indices]
    v0 = all_v0[drone_indices]
    
    print(f"  OK Randomly selected {N_DRONES_TRACK} drones to track")

N_DRONES = x0.shape[0]

print(f"  Tracking with {N_DRONES} drones (fading out {N_DRONES_TOTAL - N_DRONES} others)")

# ============================================
# EXTRACT OBJECT SHAPE FROM VIDEO
# ============================================
print("\nExtracting object contours from video...")

T_frames_px, fps, frame_size, frames_rgb = track_shape_video(
    VIDEO_PATH,
    n_points=N_DRONES,
    thresh=thresh,
    invert=invert,
    blur_ksize=blur_ksize,
    max_frames=None,  # Process entire video
    min_area=min_area
)

print(f"  OK Extracted {len(T_frames_px)} frames at {fps:.1f} FPS")
print(f"  OK Video size: {frame_size[0]}x{frame_size[1]}")

# ============================================
# CONVERT TO SIMULATION COORDINATES
# ============================================
print("\nConverting to simulation coordinates...")

T_frames_sim = px_points_to_sim(T_frames_px, frame_size, sim_scale=sim_scale)

print("  OK Coordinates converted")

# ============================================
# TEMPORAL SMOOTHING
# ============================================
print(f"\nApplying temporal smoothing (window={smooth_win})...")

T_frames_sim = smooth_targets(T_frames_sim, win=smooth_win)

print("  OK Smoothing applied")

# ============================================
# SIMULATE SHAPE TRACKING
# ============================================
print("\nSimulating shape tracking (this may take a few minutes)...")

settling_time = 3.0  # Seconds for drones to settle on first frame

traj3, tgrid3 = simulate_shape_tracking(
    x0=x0,
    v0=v0,
    T_frames_sim=T_frames_sim,
    fps=fps,
    params=params,
    dt=dt_sim,
    settling_time=settling_time
)

print(f"  OK Simulation completed: {traj3.shape[0]} timesteps")

# ============================================
# VALIDATE & SAVE
# ============================================
print("\nValidating solution...")

metrics = validate_shape_tracking(
    traj3, T_frames_sim, fps, params["rsafe"], dt_sim, verbose=True
)

# Save trajectories
output_file = "outputs/sub3/trajectories_subproblem3.npy"
np.save(output_file, traj3)
print(f"Saved trajectories: {output_file}")

# Plot metrics
plot_tracking_metrics(traj3, T_frames_sim, fps, dt_sim)

# ============================================
# VISUALIZE
# ============================================
print("\nCreating visualization...")

anim = animate_swarm(
    traj3,
    all_x0=all_x0,
    drone_indices=drone_indices,
    T_frames=T_frames_sim,
    frames_rgb=frames_rgb,
    fps=fps,
    dt=dt_sim,
    interval_ms=25,
    settling_time=settling_time,
    save_mp4=False
)

print("\n" + "="*60)
print("SUB-PROBLEM 3 COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nTuning tips:")
print("- If tracking is slow: increase kp or vmax")
print("- If drones jitter: increase smooth_win or decrease kp")
print("- If shape is wrong: adjust thresh or invert")
print("- If collisions occur: increase krep or decrease rsafe")
print("="*60)
        