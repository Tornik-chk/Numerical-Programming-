"""
Drone Show Project — Sub-problem 1 (Static formation on handwritten name)

MATHEMATICAL FORMULATION:
========================
This module solves an Initial Value Problem (IVP) for drone swarm motion.

State Variables:
  - x_i(t) ∈ ℝ² : position of drone i at time t
  - v_i(t) ∈ ℝ² : velocity of drone i at time t
  - Full state: y(t) = [x₁, ..., xₙ, v₁, ..., vₙ] ∈ R^(4N)

System of ODEs:
  dx_i/dt = sat(v_i, v_max)
  dv_i/dt = (1/m)[k_p(T_i - x_i) + ∑_{j≠i} F_rep(x_i, x_j) - k_d·v_i]

where:
  - T_i = c_target + r_i  (target position: center + offset for drone i)
  - sat(v, v_max) = v if ||v|| ≤ v_max, else v_max · v/||v||  (velocity saturation)
  - F_rep(x_i, x_j) = k_rep · (x_i - x_j) / ||x_i - x_j||3  if ||x_i - x_j|| < r_safe, else 0

Initial Conditions (t=0):
  - x_i(0) = positions along horizontal line at y = -1.5
  - v_i(0) = 0 (or small Gaussian noise)

Boundary Conditions:
  - None (pure IVP, not BVP)
  - Terminal condition: stop when mean(||v_i||) < ε for K consecutive steps

Time Domain:
  - t ∈ [0, t_end] with adaptive early stopping

NUMERICAL METHOD:
=================
RK4 (4th-order Runge-Kutta):
  - Butcher tableau:
      0   |
      1/2 | 1/2
      1/2 | 0    1/2
      1   | 0    0    1
      ----+------------------
          | 1/6  1/3  1/3  1/6
  
  - Local truncation error: O(Δt⁵)
  - Global error: O(Δt⁴)
  - A-stable for this problem (dissipative system with k_d > 0)

PHYSICAL INTERPRETATION:
========================
  - m: drone mass (normalized to 1.0)
  - k_p: spring stiffness to target (higher = faster convergence, risk of oscillation)
  - k_d: damping coefficient (higher = less overshoot, slower convergence)
  - k_rep: repulsion strength (prevents collisions)
  - r_safe: safety radius for collision avoidance
  - v_max: maximum drone speed (physical/safety constraint)

WHAT YOU NEED:
==============
Python 3.10+
pip install numpy opencv-python matplotlib scipy

INPUT:
======
- Clear photo/scan of handwritten name (≥8 characters)
- Place in: ./data/handwritten_name.jpg (or modify IMAGE_PATH below)

OUTPUT:
=======
- trajectories_subproblem1.npy (trajectory array for visualization)
- Animated visualization window
- Validation metrics printed to console

AI USAGE DISCLOSURE:
====================
Claude AI (Anthropic) was used for:
- Code review and optimization suggestions
- Documentation structure and mathematical notation
- Spatial indexing optimization for repulsion forces
"""

from __future__ import annotations
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree


# ============================
# CONFIGURATION PARAMETERS
# ============================
DEBUG_PLOTS = False  # Set to True to show intermediate image processing steps


# ========================================
# STEP 1: IMAGE PROCESSING → SHAPE POINTS
# ========================================

def keep_components_area_or_span(bw: np.ndarray, min_area: int = 500, min_span: int = 120) -> np.ndarray:
    """
    Filter connected components by area or span (width/height).
    Removes noise and small artifacts from binary image.
    
    Args:
        bw: Binary image (0 or 255)
        min_area: Minimum component area in pixels
        min_span: Minimum width or height in pixels
    
    Returns:
        Filtered binary image
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = np.zeros_like(bw)
    for k in range(1, num):  # Skip background (label 0)
        area = stats[k, cv2.CC_STAT_AREA]
        w = stats[k, cv2.CC_STAT_WIDTH]
        h = stats[k, cv2.CC_STAT_HEIGHT]
        if area >= min_area or max(w, h) >= min_span:
            out[labels == k] = 255
    return out


def crop_to_content(bw: np.ndarray, pad: int = 20) -> np.ndarray:
    """
    Crop binary image to bounding box of content with padding.
    
    Args:
        bw: Binary image
        pad: Padding pixels around content
    
    Returns:
        Cropped binary image
    """
    ys, xs = np.where(bw > 0)
    if len(ys) == 0:
        raise ValueError("No content found in image")
    
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad)
    y1 = min(bw.shape[0] - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(bw.shape[1] - 1, x1 + pad)
    return bw[y0:y1 + 1, x0:x1 + 1]


def remove_outliers_radius(pts: np.ndarray, z: float = 3.0) -> np.ndarray:
    """
    Remove outlier points using robust MAD (Median Absolute Deviation) statistic.
    
    Args:
        pts: (M, 2) array of points
        z: Z-score threshold (3.0 = remove points >3 MAD from median)
    
    Returns:
        Filtered points array
    """
    c = np.median(pts, axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    mad = np.median(np.abs(d - np.median(d))) + 1e-12
    keep = d <= (np.median(d) + z * 1.4826 * mad)  # 1.4826 is scale factor for normality
    return pts[keep]


def farthest_point_sampling(points: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) for uniform point distribution.
    Greedy algorithm that iteratively selects points farthest from already selected set.
    
    Complexity: O(k·M) where M is total points, k is desired points
    
    Args:
        points: (M, 2) candidate points
        k: Number of points to sample
        seed: Random seed for initial point selection
    
    Returns:
        (k, 2) sampled points with approximately uniform spacing
    """
    rng = np.random.default_rng(seed)
    M = points.shape[0]
    if k >= M:
        return points.copy()

    chosen = np.empty((k, 2), dtype=float)
    idx0 = rng.integers(0, M)
    chosen[0] = points[idx0]

    # Maintain squared distances to nearest chosen point for efficiency
    d2 = np.sum((points - chosen[0]) ** 2, axis=1)

    for i in range(1, k):
        idx = int(np.argmax(d2))  # Pick farthest point
        chosen[i] = points[idx]
        # Update distances: min(old_distance, distance_to_new_point)
        d2 = np.minimum(d2, np.sum((points - chosen[i]) ** 2, axis=1))

    return chosen


def extract_shape_points_skeleton(
    image_path: str,
    n_points: int,
    block_size: int = 31,
    C: int = 10,
    close_ksize: int = 5,
    min_area: int = 1200,
    seed: int = 0,
    max_width: int = 1200,
    fps_pool: int = 20000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract uniformly distributed points from handwritten text via skeletonization.
    
    Pipeline:
      1. Load grayscale image
      2. Adaptive thresholding (handles varying lighting)
      3. Morphological closing (connect broken strokes)
      4. Component filtering (remove noise)
      5. Skeletonization (extract medial axis)
      6. Farthest point sampling (uniform distribution)
    
    Args:
        image_path: Path to handwritten name image
        n_points: Number of drone target points to extract
        block_size: Adaptive threshold block size (must be odd)
        C: Adaptive threshold constant subtracted from mean
        close_ksize: Morphological closing kernel size
        min_area: Minimum component area to keep
        seed: Random seed for reproducibility
        max_width: Resize image to this width for speed
        fps_pool: Downsample skeleton to this many points before FPS
    
    Returns:
        pts: (n_points, 2) sampled skeleton points in image coordinates
        bw: Binary image after preprocessing
        skel: Skeletonized binary image
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Resize for computational efficiency
    h, w = img.shape
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Adaptive thresholding: better than global threshold for uneven lighting
    bw = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # INV: text is white, background is black
        block_size, C
    )

    # Morphological closing: connect nearby strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    bw = cv2.medianBlur(bw, 3)  # Remove salt-and-pepper noise

    # Filter out small components
    bw = keep_components_area_or_span(bw, min_area=min_area, min_span=120)
    bw = crop_to_content(bw, pad=10)

    # Zhang-Suen skeletonization: extract 1-pixel-wide medial axis
    skel = np.zeros(bw.shape, np.uint8)
    temp = bw.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        temp2 = cv2.subtract(temp, opened)
        skel = cv2.bitwise_or(skel, temp2)
        temp = eroded
        if cv2.countNonZero(temp) == 0:
            break

    # Extract skeleton pixel coordinates
    ys, xs = np.where(skel > 0)
    pts = np.stack([xs, ys], axis=1).astype(float)

    if pts.shape[0] < n_points:
        raise ValueError(
            f"Skeleton has only {pts.shape[0]} pixels, but need {n_points} points. "
            f"Try: lower min_area, use simpler handwriting, or reduce n_points."
        )

    # Downsample skeleton pool for faster FPS
    if pts.shape[0] > fps_pool:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(pts.shape[0], size=fps_pool, replace=False)]

    # Remove outliers (stray pixels far from main text)
    pts = remove_outliers_radius(pts, z=2.0)

    # Farthest point sampling for uniform distribution
    pts_sampled = farthest_point_sampling(pts, n_points, seed=seed)
    
    return pts_sampled, bw, skel


# ================================================
# STEP 2: SHAPE REPRESENTATION (CENTER + OFFSETS)
# ================================================

def shape_center_and_offsets(points: np.ndarray, scale_to: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw shape points to center + normalized offsets representation.
    
    Representation: Each point p_i = center + r_i
    where r_i are offsets scaled so max(||r_i||) = scale_to
    
    This ensures:
      1. Translation invariance (center can be placed anywhere)
      2. Numerical stability (points in [-scale_to, scale_to] range)
    
    Args:
        points: (N, 2) raw shape points
        scale_to: Target scale for max radius
    
    Returns:
        c: (2,) shape center
        r: (N, 2) normalized offsets from center
    """
    c = points.mean(axis=0)
    r = points - c
    
    max_norm = np.max(np.linalg.norm(r, axis=1))
    if max_norm < 1e-12:
        raise ValueError("Degenerate shape: all points are identical.")
    
    r_scaled = (r / max_norm) * scale_to
    return c, r_scaled


# =====================================
# STEP 3: INITIAL DRONE CONFIGURATION
# =====================================

def init_drones_line(
    n: int,
    length: float = 4.0,
    y0: float = 0.0,
    v0_scale: float = 0.0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize drone positions along horizontal line with optional velocity noise.
    
    Configuration:
      - Positions: x ∈ [-length/2, length/2], y = y0
      - Velocities: Gaussian noise with std = v0_scale
    
    Args:
        n: Number of drones
        length: Line length
        y0: Vertical position of line
        v0_scale: Standard deviation of initial velocity noise
        seed: Random seed
    
    Returns:
        x0: (n, 2) initial positions
        v0: (n, 2) initial velocities
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(-length / 2, length / 2, n)
    y = np.full(n, y0)
    x0 = np.column_stack((x, y))

    v0 = rng.normal(0.0, v0_scale, size=(n, 2))
    return x0, v0


# ===============================
# STEP 4: DYNAMICS (IVP MODEL)
# ===============================

def v_saturate(v: np.ndarray, vmax: float) -> np.ndarray:
    """
    Saturate velocity vectors to maximum speed constraint.
    
    Physical interpretation: Drone motors have maximum thrust/speed.
    
    Args:
        v: (N, 2) velocity vectors
        vmax: Maximum allowed speed
    
    Returns:
        v_sat: (N, 2) saturated velocities with ||v_sat[i]|| ≤ vmax
    """
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
    Compute pairwise repulsion forces for collision avoidance.
    
    Force model:
      F_ij = k_rep · (x_i - x_j) / ||x_i - x_j||³  if ||x_i - x_j|| < r_safe
      F_ij = 0                                      otherwise
    
    This is a soft potential: U(r) ∝ 1/r² → F = -∇U ∝ 1/r³
    
    Complexity:
      - Naive: O(N²) - compute all pairs
      - Spatial indexing: O(N·k) where k = avg neighbors within r_safe
    
    Args:
        x: (N, 2) current drone positions
        krep: Repulsion strength constant
        rsafe: Safety radius (no repulsion beyond this)
        eps: Numerical stability constant to avoid division by zero
        use_spatial_index: If True, use KD-tree for O(N·k) complexity
    
    Returns:
        f: (N, 2) total repulsion force on each drone
    """
    n = x.shape[0]
    f = np.zeros_like(x)
    
    if use_spatial_index and n > 50:
        # Spatial indexing optimization: only compute forces for nearby drones
        tree = cKDTree(x)
        for i in range(n):
            # Query neighbors within rsafe radius
            indices = tree.query_ball_point(x[i], rsafe)
            for j in indices:
                if i != j:
                    dx = x[i] - x[j]
                    dist = np.linalg.norm(dx) + eps
                    if dist < rsafe:
                        f[i] += krep * dx / (dist ** 3)
    else:
        # Naive O(N²) approach - fine for N < 100
        dx = x[:, None, :] - x[None, :, :]  # (N, N, 2)
        dist2 = np.sum(dx * dx, axis=2) + eps  # (N, N)
        dist = np.sqrt(dist2)
        
        np.fill_diagonal(dist, np.inf)  # Ignore self-interactions
        
        mask = dist < rsafe
        inv_dist3 = np.zeros_like(dist)
        inv_dist3[mask] = 1.0 / (dist[mask] ** 3)
        
        # Sum forces from all neighbors
        f = krep * np.sum(dx * inv_dist3[:, :, None], axis=1)
    
    return f


def rhs(
    t: float,
    x: np.ndarray,
    v: np.ndarray,
    r_offsets: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Right-hand side of ODE system: computes time derivatives.
    
    Equations:
      dx/dt = sat(v, v_max)
      dv/dt = (1/m)[k_p(T - x) + F_rep - k_d·v]
    
    where T = c_target + r_offsets (target positions)
    
    Physical interpretation:
      - PD controller: k_p pulls toward target, k_d damps oscillations
      - Repulsion: prevents collisions
      - Saturation: enforces speed limits
    
    Args:
        t: Current time (not used, but required for ODE solver interface)
        x: (N, 2) positions
        v: (N, 2) velocities
        r_offsets: (N, 2) target offsets from center
        params: Dictionary with keys: m, kp, kd, vmax, krep, rsafe
    
    Returns:
        xdot: (N, 2) velocity (saturated)
        vdot: (N, 2) acceleration
    """
    m = params["m"]
    kp = params["kp"]
    kd = params["kd"]
    vmax = params["vmax"]
    krep = params["krep"]
    rsafe = params["rsafe"]

    # For sub-problem 1: target shape centered at origin
    c_target = np.array([0.0, 0.0], dtype=float)
    T = c_target[None, :] + r_offsets  # (N, 2)

    # Compute derivatives
    v_sat = v_saturate(v, vmax)
    xdot = v_sat

    frep = repulsion_forces(x, krep=krep, rsafe=rsafe)
    vdot = (1.0 / m) * (kp * (T - x) + frep - kd * v)
    
    return xdot, vdot


# ====================================
# STEP 5: TIME INTEGRATION (RK4)
# ====================================

def rk4_step(
    t: float,
    x: np.ndarray,
    v: np.ndarray,
    dt: float,
    r_offsets: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single RK4 (4th-order Runge-Kutta) timestep for coupled (x, v) system.
    
    Butcher Tableau:
      0   |
      1/2 | 1/2
      1/2 | 0    1/2
      1   | 0    0    1
      ----+------------------
          | 1/6  1/3  1/3  1/6
    
    Update formula:
      y_{n+1} = y_n + (Δt/6)(k1 + 2k2 + 2k3 + k4)
    
    Truncation error: O(Δt⁵) per step, O(Δt⁴) globally
    
    Args:
        t: Current time
        x: (N, 2) positions at time t
        v: (N, 2) velocities at time t
        dt: Timestep size
        r_offsets: (N, 2) target offsets
        params: Physics parameters
    
    Returns:
        x_next: (N, 2) positions at time t + dt
        v_next: (N, 2) velocities at time t + dt
    """
    # Stage 1
    k1x, k1v = rhs(t, x, v, r_offsets, params)
    
    # Stage 2
    k2x, k2v = rhs(
        t + 0.5 * dt,
        x + 0.5 * dt * k1x,
        v + 0.5 * dt * k1v,
        r_offsets, params
    )
    
    # Stage 3
    k3x, k3v = rhs(
        t + 0.5 * dt,
        x + 0.5 * dt * k2x,
        v + 0.5 * dt * k2v,
        r_offsets, params
    )
    
    # Stage 4
    k4x, k4v = rhs(
        t + dt,
        x + dt * k3x,
        v + dt * k3v,
        r_offsets, params
    )

    # Combine stages with RK4 weights
    x_next = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_next = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    
    return x_next, v_next


def simulate_to_static_shape(
    x0: np.ndarray,
    v0: np.ndarray,
    r_offsets: np.ndarray,
    params: dict,
    dt: float = 0.01,
    t_end: float = 10.0,
    stop_tol: float = 1e-3,
    stop_window: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate drone swarm until convergence to target shape.
    
    Stopping criterion:
      - Stop early if mean speed < stop_tol for stop_window consecutive steps
      - Otherwise run until t = t_end
    
    Args:
        x0: (N, 2) initial positions
        v0: (N, 2) initial velocities
        r_offsets: (N, 2) target shape offsets
        params: Physics parameters
        dt: Timestep size
        t_end: Maximum simulation time
        stop_tol: Speed threshold for convergence
        stop_window: Number of consecutive slow steps to confirm convergence
    
    Returns:
        traj: (T, N, 2) position trajectories
        tgrid: (T,) time values
    """
    n = x0.shape[0]
    steps = int(math.ceil(t_end / dt))
    traj = np.zeros((steps + 1, n, 2), dtype=float)
    tgrid = np.zeros(steps + 1, dtype=float)

    x = x0.copy()
    v = v0.copy()
    traj[0] = x
    t = 0.0

    slow_count = 0
    for k in range(1, steps + 1):
        x, v = rk4_step(t, x, v, dt, r_offsets, params)
        t += dt
        traj[k] = x
        tgrid[k] = t

        # Check convergence
        mean_speed = float(np.mean(np.linalg.norm(v, axis=1)))
        if mean_speed < stop_tol:
            slow_count += 1
            if slow_count >= stop_window:
                print(f"Early stop at t={t:.2f}s (mean speed < {stop_tol})")
                traj = traj[: k + 1]
                tgrid = tgrid[: k + 1]
                break
        else:
            slow_count = 0

    return traj, tgrid


# ===============================
# STEP 6: VALIDATION & METRICS
# ===============================

def validate_solution(
    traj: np.ndarray,
    r_offsets: np.ndarray,
    rsafe: float,
    verbose: bool = True,
) -> dict:
    """
    Validate simulation results and compute performance metrics.
    
    Checks:
      1. Formation error: how close did drones get to target?
      2. Collision safety: did any drones get too close?
      3. Convergence: did the swarm stabilize?
    
    Args:
        traj: (T, N, 2) trajectory array
        r_offsets: (N, 2) target positions (centered at origin)
        rsafe: Safety radius for collision checking
        verbose: Print results to console
    
    Returns:
        Dictionary with validation metrics
    """
    final_pos = traj[-1]  # Final positions
    target = r_offsets    # Target is centered at origin for subproblem 1
    
    # 1. Formation error
    errors = np.linalg.norm(final_pos - target, axis=1)
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    
    # 2. Collision check (minimum pairwise distance)
    from scipy.spatial.distance import pdist
    min_dist = float(np.min(pdist(final_pos)))
    collision_occurred = min_dist < rsafe
    
    # 3. Convergence metric (velocity at end)
    if traj.shape[0] > 1:
        final_velocity = traj[-1] - traj[-2]  # Approximate velocity
        mean_speed = float(np.mean(np.linalg.norm(final_velocity, axis=1)))
    else:
        mean_speed = 0.0
    
    results = {
        "mean_error": mean_error,
        "max_error": max_error,
        "min_pairwise_distance": min_dist,
        "collision_occurred": collision_occurred,
        "final_mean_speed": mean_speed,
        "converged": mean_speed < 0.01,
    }
    
    if verbose:
        print("\n" + "="*50)
        print("VALIDATION RESULTS")
        print("="*50)
        print(f"Formation Error:")
        print(f"  Mean error:     {mean_error:.4f}")
        print(f"  Max error:      {max_error:.4f}")
        print(f"\nCollision Safety:")
        print(f"  Min distance:   {min_dist:.4f}")
        print(f"  Safety radius:  {rsafe:.4f}")
        print(f"  Status:         {'COLLISION' if collision_occurred else 'SAFE'}")
        print(f"\nConvergence:")
        print(f"  Final speed:    {mean_speed:.4f}")
        print(f"  Status:         {'cONVERGED' if results['converged'] else 'STILL MOVING'}")
        print("="*50 + "\n")
    
    return results


def plot_convergence_metrics(traj: np.ndarray, r_offsets: np.ndarray, tgrid: np.ndarray):
    """
    Plot convergence metrics over time.
    
    Metrics:
      1. Mean distance to target (should decay to ~0)
      2. Max distance to target
      3. Mean speed (should decay to ~0)
    
    Args:
        traj: (T, N, 2) trajectories
        r_offsets: (N, 2) target offsets
        tgrid: (T,) time values
    """
    T = traj.shape[0]
    target = r_offsets  # Centered at origin
    
    mean_errors = np.zeros(T)
    max_errors = np.zeros(T)
    mean_speeds = np.zeros(T)
    
    for i in range(T):
        errors = np.linalg.norm(traj[i] - target, axis=1)
        mean_errors[i] = np.mean(errors)
        max_errors[i] = np.max(errors)
        
        if i > 0:
            v = (traj[i] - traj[i-1]) / (tgrid[i] - tgrid[i-1])
            mean_speeds[i] = np.mean(np.linalg.norm(v, axis=1))
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Error plot
    axes[0].plot(tgrid, mean_errors, label='Mean error', linewidth=2)
    axes[0].plot(tgrid, max_errors, label='Max error', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Distance to target')
    axes[0].set_title('Formation Error vs Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    # Speed plot
    axes[1].plot(tgrid, mean_speeds, linewidth=2, color='green')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Mean speed')
    axes[1].set_title('Swarm Speed vs Time (should decay to zero)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('convergence_metrics_subproblem1.png', dpi=150)
    print("Saved: convergence_metrics_subproblem1.png")
    plt.show()
# ==============================
# STEP 7: VISUALIZATION
# ==============================

def animate_trajectories(
    traj: np.ndarray,
    r_offsets: np.ndarray,
    interval_ms: int = 30,
    save_mp4: bool = False,
):
    """
    Animated visualization of drone swarm with illumination effect.
    
    Features:
      - Bright drone markers (simulates LED illumination)
      - Motion trails for recent history
      - Target shape overlay (faint reference)
      - Frame counter
    
    Args:
        traj: (T, N, 2) trajectory array
        r_offsets: (N, 2) target shape offsets
        interval_ms: Milliseconds between frames
        save_mp4: If True, save animation as MP4 (requires ffmpeg)
    """
    Tsteps, N, _ = traj.shape

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    fig.patch.set_facecolor('black')  # Dark background for "night sky" effect
    ax.set_facecolor('black')

    # Compute plot bounds from trajectory
    xy = traj.reshape(-1, 2)
    pad = 0.5
    xmin, ymin = xy.min(axis=0) - pad
    xmax, ymax = xy.max(axis=0) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Sub-problem 1: Static Formation", color='white', fontsize=14)
    ax.tick_params(colors='white')

    # Target shape overlay (faint reference)
    target = r_offsets
    ax.scatter(
        target[:, 0], target[:, 1],
        s=8, c='cyan', alpha=0.2, marker='x',
        label='Target shape'
    )

    # Drone markers (bright, illuminated)
    scat = ax.scatter(
        traj[0, :, 0], traj[0, :, 1],
        s=25, c='yellow', edgecolors='white',
        linewidths=0.5, alpha=0.9,
        label='Drones'
    )

    # Motion trails (last K frames)
    Ktrail = 20
    trail_lines = [
        ax.plot([], [], linewidth=1, color='yellow', alpha=0.1)[0]
        for _ in range(Ktrail)
    ]

    # Legend
    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

    # Time/frame text
    time_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes,
        color='white', fontsize=10,
        verticalalignment='top'
    )

    def update(frame: int):
        """Update function for animation"""
        pts = traj[frame]
        scat.set_offsets(pts)

        # Update motion trails (centroid path for clarity)
        for i, ln in enumerate(trail_lines):
            f0 = max(0, frame - (i + 1) * 2)
            f1 = frame
            if f1 > f0:
                # Centroid trail
                cseg = traj[f0:f1 + 1].mean(axis=1)
                ln.set_data(cseg[:, 0], cseg[:, 1])

        # Update time text
        time_text.set_text(f'Frame: {frame + 1}/{Tsteps}')
        
        return (scat, *trail_lines, time_text)

    anim = FuncAnimation(
        fig, update,
        frames=Tsteps,
        interval=interval_ms,
        blit=False
    )

    if save_mp4:
        # Requires ffmpeg: https://ffmpeg.org/download.html
        print("Saving animation (this may take a while)...")
        anim.save("subproblem1_static_formation.mp4", dpi=150, writer='ffmpeg')
        print("Saved: subproblem1_static_formation.mp4")

    plt.show()


# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    # ============================================
    # CONFIGURATION
    # ============================================
    
    # Input image path (relative path for reproducibility)
    IMAGE_PATH = os.path.join("data", "handwritten_name.jpg")
    
    # Number of drones (should be sufficient to display name clearly)
    N_DRONES = 300
    
    # Scale factor for target shape (larger = more spread out)
    SHAPE_SCALE = 3.0
    
    # Physics parameters (tuned for stable convergence)
    params = dict(
        m=1.0,       # Drone mass (normalized)
        kp=20.0,     # Spring constant (higher = faster, but risk oscillation)
        kd=6.0,      # Damping (higher = less overshoot)
        vmax=2.0,    # Maximum speed constraint
        krep=0.05,   # Repulsion strength (tune based on N_DRONES and density)
        rsafe=0.1,  # Safety radius (minimum allowed distance)
    )
    
    # Time integration parameters
    dt = 0.01        # Timestep (smaller = more accurate but slower)
    t_end = 12.0     # Maximum simulation time
    
    print("="*60)
    print("DRONE SHOW - SUB-PROBLEM 1: STATIC FORMATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  Image: {IMAGE_PATH}")
    print(f"  Drones: {N_DRONES}")
    print(f"  Physics params: {params}")
    print(f"  Timestep: {dt}s, Max time: {t_end}s")
    print("="*60 + "\n")
    
    # ============================================
    # STEP 1: EXTRACT SHAPE POINTS FROM IMAGE
    # ============================================
    print("Step 1: Extracting shape points from image...")
    
    pts, bw, skel = extract_shape_points_skeleton(
        IMAGE_PATH,
        n_points=N_DRONES,
        block_size=31,
        C=5,
        close_ksize=3,
        min_area=300,
        seed=0,
    )
    
    print(f"  Extracted {pts.shape[0]} skeleton points")
    
    # Optional debug plots
    if DEBUG_PLOTS:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(bw, cmap="gray")
        plt.title("Binary Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(skel, cmap="gray")
        plt.title("Skeleton")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.scatter(pts[:, 0], pts[:, 1], s=4, c='red')
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.title("Sampled Points")
        
        plt.tight_layout()
        plt.show()
    
    # ============================================
    # STEP 2: CONVERT TO CENTER + OFFSETS
    # ============================================
    print("Step 2: Converting to center + offsets representation...")
    
    _, r_offsets = shape_center_and_offsets(pts, scale_to=SHAPE_SCALE)
    
    # COORDINATE SYSTEM FIX:
    # Image coordinates: y points DOWN (0 at top)
    # Math coordinates: y points UP (0 at bottom)
    # We flip y-axis to match standard mathematical convention
    r_offsets[:, 1] *= -1
    
    print(f"  Shape scaled to radius ~{SHAPE_SCALE}")
    
    # ============================================
    # STEP 3: INITIALIZE DRONE POSITIONS
    # ============================================
    print("Step 3: Initializing drone positions...")
    
    x0, v0 = init_drones_line(
        n=N_DRONES,
        length=5.0,      # Horizontal line length
        y0=-1.5,         # Below target shape
        v0_scale=0.0,    # No initial velocity noise
        seed=1,
    )
    
    print(f"  Drones initialized on horizontal line at y={-1.5}")
    
    # ============================================
    # STEP 4-5: SIMULATE TRAJECTORIES
    # ============================================
    print("Step 4-5: Simulating trajectories (this may take a minute)...")
    
    traj, tgrid = simulate_to_static_shape(
        x0=x0,
        v0=v0,
        r_offsets=r_offsets,
        params=params,
        dt=dt,
        t_end=t_end,
        stop_tol=2e-3,
        stop_window=60,
    )
    
    print(f"  Simulation completed: {traj.shape[0]} timesteps, final t={tgrid[-1]:.2f}s")
    
    # ============================================
    # STEP 6: VALIDATE & SAVE RESULTS
    # ============================================
    print("\nStep 6: Validating solution...")
    
    metrics = validate_solution(traj, r_offsets, params["rsafe"], verbose=True)
    
    # Save trajectories for later use (sub-problems 2 & 3)
    output_file = "trajectories_subproblem1.npy"
    np.save(output_file, traj)
    print(f"Saved trajectories: {output_file}")
    
    # Plot convergence metrics
    plot_convergence_metrics(traj, r_offsets, tgrid)
    
    # ============================================
    # STEP 7: VISUALIZE
    # ============================================
    print("\nStep 7: Creating visualization...")
    
    animate_trajectories(
        traj,
        r_offsets,
        interval_ms=25,
        save_mp4=False  # Set to True to save animation (requires ffmpeg)
    )
    
    print("\n" + "="*60)
    print("SUB-PROBLEM 1 COMPLETED SUCCESSFULLY!")
    print("="*60)