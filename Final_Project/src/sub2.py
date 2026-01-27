"""
Drone Show Project — Sub-problem 2 (Transition to "Happy New Year!")

MATHEMATICAL FORMULATION:
========================
This module solves a time-varying target IVP for smooth formation transitions.

State Variables:
  - x_i(t) ∈ ℝ² : position of drone i at time t
  - v_i(t) ∈ ℝ² : velocity of drone i at time t

System of ODEs (same as Sub-problem 1, but with time-varying targets):
  dx_i/dt = sat(v_i, v_max)
  dv_i/dt = (1/m)[k_p(T_i(t) - x_i) + ∑_{j≠i} F_rep(x_i, x_j) - k_d·v_i]

Time-Varying Target (Linear Interpolation):
  T_i(t) = (1 - a(t))·T0_i + a(t)·T1_i
  
  where a(t) = min(1, t/T_ramp) is a ramp function:
    - a(0) = 0 → T_i(0) = T0_i (handwritten name)
    - a(t ≥ T_ramp) = 1 → T_i(t) = T1_i ("Happy New Year!")
  
  This ensures smooth transition without sudden jumps in target positions.

Initial Conditions (t=0):
  - x_i(0) = final positions from Sub-problem 1
  - v_i(0) = final velocities from Sub-problem 1 (approximate from trajectory)

Assignment Strategy:
  - Greedy nearest-neighbor matching between name points and greeting points
  - Minimizes trajectory crossings and total travel distance
  - Complexity: O(N²) but acceptable for N ~ 300

NUMERICAL METHOD:
=================
Same RK4 implementation as Sub-problem 1, but with time-varying target T(t).

PHYSICAL INTERPRETATION:
========================
The transition simulates synchronized formation change:
  - T_ramp controls transition speed (longer = smoother)
  - Same collision avoidance as Sub-problem 1
  - Velocity continuity from previous formation

REQUIREMENTS:
=============
pip install numpy opencv-python matplotlib scipy

INPUT:
======
- trajectories_subproblem1.npy (output from Sub-problem 1)
- Same handwritten name image from Sub-problem 1
- Same N_DRONES parameter

OUTPUT:
=======
- trajectories_subproblem2.npy (trajectory array)
- greeting.png (rendered "Happy New Year!" text)
- Animated visualization

AI USAGE DISCLOSURE:
====================
Claude AI (Anthropic) was used for:
- Code structure and optimization
- Documentation and mathematical formulation
- Target assignment algorithm selection
- Validation metrics implementation
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
# IMAGE PROCESSING (reused from Sub-1)
# ========================================

def keep_components_area_span(
    bw: np.ndarray,
    min_area: int = 300,
    min_span: int = 120,
    min_dot_area: int = 20
) -> np.ndarray:
    """
    Filter connected components by area/span.
    Special handling for dots (exclamation points, periods).
    
    Args:
        bw: Binary image
        min_area: Minimum area for regular components
        min_span: Minimum width or height
        min_dot_area: Minimum area for dot-like components (lower = captures smaller dots)
    
    Returns:
        Filtered binary image
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = np.zeros_like(bw)

    for k in range(1, num):
        area = stats[k, cv2.CC_STAT_AREA]
        w = stats[k, cv2.CC_STAT_WIDTH]
        h = stats[k, cv2.CC_STAT_HEIGHT]

        # Detect dots: small span but non-trivial area (even very small dots)
        is_dot = (area >= min_dot_area) and (w <= 30) and (h <= 30)

        if area >= min_area or max(w, h) >= min_span or is_dot:
            out[labels == k] = 255

    return out


def crop_to_content(bw: np.ndarray, pad: int = 20) -> np.ndarray:
    """
    Crop to content bounding box with padding.
    Also clears thin border to remove edge artifacts.
    """
    ys, xs = np.where(bw > 0)
    if len(ys) == 0:
        raise ValueError("No content in image")
    
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad)
    y1 = min(bw.shape[0] - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(bw.shape[1] - 1, x1 + pad)
    
    out = bw[y0:y1 + 1, x0:x1 + 1].copy()
    
    # Clear thin borders to avoid stray pixels
    out[:3, :] = 0
    out[-3:, :] = 0
    out[:, :3] = 0
    out[:, -3:] = 0
    
    return out


def farthest_point_sampling(points: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    Farthest Point Sampling for uniform distribution.
    Same implementation as Sub-problem 1.
    """
    rng = np.random.default_rng(seed)
    M = points.shape[0]
    if k >= M:
        return points.copy()

    chosen = np.empty((k, 2), dtype=float)
    idx0 = int(rng.integers(0, M))
    chosen[0] = points[idx0]

    d2 = np.sum((points - chosen[0]) ** 2, axis=1)
    for i in range(1, k):
        idx = int(np.argmax(d2))
        chosen[i] = points[idx]
        d2 = np.minimum(d2, np.sum((points - chosen[i]) ** 2, axis=1))

    return chosen


def extract_shape_points_skeleton(
    image_path: str,
    n_points: int,
    block_size: int = 31,
    C: int = 5,
    close_ksize: int = 3,
    min_area: int = 300,
    min_span: int = 120,
    seed: int = 0,
    max_width: int = 1200,
    fps_pool: int = 20000,
    min_dot_area: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract uniformly distributed points from text image via skeletonization.
    Same as Sub-problem 1, works for both handwritten and rendered text.
    
    Returns:
        pts_sampled: (n_points, 2) sampled skeleton points
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

    # Adaptive thresholding
    bw = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    bw = cv2.medianBlur(bw, 3)

    # Filter components (with dot handling for exclamation marks)
    bw = keep_components_area_span(bw, min_area=min_area, min_span=min_span, min_dot_area=min_dot_area)
    bw = crop_to_content(bw, pad=25)

    # Zhang-Suen skeletonization
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

    # Extract skeleton pixels
    ys, xs = np.where(skel > 0)
    pts = np.stack([xs, ys], axis=1).astype(float)

    if pts.shape[0] < n_points:
        raise ValueError(
            f"Skeleton has only {pts.shape[0]} pixels, need {n_points}. "
            f"Try: reduce n_points, increase image size, or adjust thresholds."
        )

    # Downsample pool before FPS
    if pts.shape[0] > fps_pool:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(pts.shape[0], size=fps_pool, replace=False)]

    # Farthest point sampling
    pts_sampled = farthest_point_sampling(pts, n_points, seed=seed)
    return pts_sampled, bw, skel


def shape_center_and_offsets(points: np.ndarray, scale_to: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw points to center + normalized offsets.
    Same as Sub-problem 1.
    """
    c = points.mean(axis=0)
    r = points - c
    max_norm = np.max(np.linalg.norm(r, axis=1))
    if max_norm < 1e-12:
        raise ValueError("Degenerate shape: all points identical")
    r_scaled = (r / max_norm) * scale_to
    return c, r_scaled


# ========================================
# TEXT RENDERING
# ========================================

def render_text_image(
    text: str,
    out_path: str = "greeting.png",
    width: int = 1600,
    height: int = 500,
    font_scale: float = 3.2,
    thickness: int = 14,
) -> str:
    """
    Render text to image using OpenCV with proper handling for punctuation.
    
    Args:
        text: Text to render (e.g., "HAPPY NEW YEAR!")
        out_path: Output image path
        width: Image width in pixels
        height: Image height in pixels
        font_scale: Font size multiplier
        thickness: Text stroke thickness
    
    Returns:
        Path to saved image
    """
    # White background
    img = np.full((height, width), 255, dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_DUPLEX  # Better font for punctuation
    
    # Calculate text size for centering
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Center text (adjust for baseline)
    x = max(10, (width - tw) // 2)
    y = max(th + 10, (height + th) // 2)
    
    # Draw black text on white background
    cv2.putText(img, text, (x, y), font, font_scale, (0,), thickness, cv2.LINE_AA)
    
    cv2.imwrite(out_path, img)
    print(f"Rendered text image: {out_path}")
    return out_path


# ========================================
# TARGET ASSIGNMENT (minimize crossings)
# ========================================

def greedy_match_targets(x_start: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Greedy nearest-neighbor assignment to minimize trajectory crossings.
    
    Algorithm:
      For each drone i in order:
        Assign the closest unassigned target point
    
    This is NOT optimal (optimal requires Hungarian algorithm O(N³)),
    but gives good results for N~300 and runs in O(N²).
    
    Args:
        x_start: (N, 2) current drone positions
        targets: (N, 2) target points (unordered)
    
    Returns:
        (N, 2) reordered targets, one per drone
    """
    N = x_start.shape[0]
    remaining = targets.copy()
    assigned = np.zeros_like(targets)

    for i in range(N):
        # Find closest remaining target to drone i
        d2 = np.sum((remaining - x_start[i]) ** 2, axis=1)
        j = int(np.argmin(d2))
        
        # Assign and remove from pool
        assigned[i] = remaining[j]
        remaining = np.delete(remaining, j, axis=0)
    
    return assigned


# ========================================
# SHAPE RESCALING
# ========================================

def rescale_to_match_bbox(
    source: np.ndarray,
    reference: np.ndarray,
    match: str = "height"
) -> np.ndarray:
    """
    Scale source shape to match reference shape's bounding box size.
    
    This ensures both formations have similar visual scale.
    
    Args:
        source: (N, 2) points to rescale
        reference: (N, 2) reference points
        match: "height" or "width" - which dimension to match
    
    Returns:
        Rescaled source points
    """
    src_min, src_max = source.min(axis=0), source.max(axis=0)
    ref_min, ref_max = reference.min(axis=0), reference.max(axis=0)

    src_w, src_h = (src_max - src_min)
    ref_w, ref_h = (ref_max - ref_min)

    if match == "height":
        scale_factor = ref_h / (src_h + 1e-12)
    elif match == "width":
        scale_factor = ref_w / (src_w + 1e-12)
    else:
        raise ValueError("match must be 'height' or 'width'")

    return source * scale_factor


# ========================================
# DYNAMICS (time-varying target)
# ========================================

def v_saturate(v: np.ndarray, vmax: float) -> np.ndarray:
    """Saturate velocity to max speed. Same as Sub-problem 1."""
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
    Pairwise repulsion forces. Same as Sub-problem 1.
    Includes spatial indexing optimization for N > 50.
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


def rhs_transition(
    t: float,
    x: np.ndarray,
    v: np.ndarray,
    T0: np.ndarray,
    T1: np.ndarray,
    params: dict,
    T_ramp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Right-hand side with time-varying target T(t).
    
    Target interpolation:
      T(t) = (1 - a(t))·T0 + a(t)·T1
      where a(t) = min(1, t/T_ramp)
    
    This creates a smooth linear transition from formation T0 to T1.
    
    Args:
        t: Current time
        x: (N, 2) positions
        v: (N, 2) velocities
        T0: (N, 2) initial formation targets
        T1: (N, 2) final formation targets
        params: Physics parameters
        T_ramp: Transition duration
    
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

    # Linear interpolation parameter: 0 → 1 over [0, T_ramp]
    alpha = min(1.0, t / T_ramp)
    
    # Time-varying target
    T = (1.0 - alpha) * T0 + alpha * T1

    # Same dynamics as Sub-problem 1, but with T(t)
    xdot = v_saturate(v, vmax)
    frep = repulsion_forces(x, krep=krep, rsafe=rsafe)
    vdot = (1.0 / m) * (kp * (T - x) + frep - kd * v)
    
    return xdot, vdot


def rk4_step_transition(
    t: float,
    x: np.ndarray,
    v: np.ndarray,
    dt: float,
    T0: np.ndarray,
    T1: np.ndarray,
    params: dict,
    T_ramp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RK4 step for transition dynamics.
    Same structure as Sub-problem 1, but calls rhs_transition.
    """
    k1x, k1v = rhs_transition(t, x, v, T0, T1, params, T_ramp)
    
    k2x, k2v = rhs_transition(
        t + 0.5 * dt,
        x + 0.5 * dt * k1x,
        v + 0.5 * dt * k1v,
        T0, T1, params, T_ramp
    )
    
    k3x, k3v = rhs_transition(
        t + 0.5 * dt,
        x + 0.5 * dt * k2x,
        v + 0.5 * dt * k2v,
        T0, T1, params, T_ramp
    )
    
    k4x, k4v = rhs_transition(
        t + dt,
        x + dt * k3x,
        v + dt * k3v,
        T0, T1, params, T_ramp
    )

    x_next = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_next = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    
    return x_next, v_next


def simulate_transition(
    x0: np.ndarray,
    v0: np.ndarray,
    T0: np.ndarray,
    T1: np.ndarray,
    params: dict,
    dt: float = 0.01,
    t_end: float = 10.0,
    T_ramp: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate transition from formation T0 to T1.
    
    Args:
        x0: (N, 2) initial positions
        v0: (N, 2) initial velocities
        T0: (N, 2) initial formation targets
        T1: (N, 2) final formation targets
        params: Physics parameters
        dt: Timestep
        t_end: Total simulation time
        T_ramp: Time for target transition (after this, T=T1 constantly)
    
    Returns:
        traj: (T, N, 2) position trajectories
        tgrid: (T,) time values
    """
    steps = int(math.ceil(t_end / dt))
    N = x0.shape[0]
    traj = np.zeros((steps + 1, N, 2), dtype=float)
    tgrid = np.zeros(steps + 1, dtype=float)

    x = x0.copy()
    v = v0.copy()
    traj[0] = x

    t = 0.0
    for k in range(1, steps + 1):
        x, v = rk4_step_transition(t, x, v, dt, T0, T1, params, T_ramp)
        t += dt
        traj[k] = x
        tgrid[k] = t

    return traj, tgrid


# ========================================
# VALIDATION
# ========================================

def validate_transition(
    traj: np.ndarray,
    T1: np.ndarray,
    rsafe: float,
    verbose: bool = True,
) -> dict:
    """
    Validate transition simulation results.
    
    Checks:
      1. Formation error at end
      2. Collision safety throughout
      3. Trajectory smoothness
    
    Args:
        traj: (T, N, 2) trajectories
        T1: (N, 2) final target positions
        rsafe: Safety radius
        verbose: Print results
    
    Returns:
        Dictionary with validation metrics
    """
    final_pos = traj[-1]
    
    # 1. Final formation error
    errors = np.linalg.norm(final_pos - T1, axis=1)
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    
    # 2. Check for collisions throughout trajectory
    min_dist_overall = float('inf')
    for t in range(traj.shape[0]):
        min_dist_t = float(np.min(pdist(traj[t])))
        min_dist_overall = min(min_dist_overall, min_dist_t)
    
    collision_occurred = min_dist_overall < rsafe
    
    # 3. Smoothness: max acceleration
    if traj.shape[0] > 2:
        vel = np.diff(traj, axis=0)  # Approximate velocities
        acc = np.diff(vel, axis=0)   # Approximate accelerations
        max_acc = float(np.max(np.linalg.norm(acc, axis=2)))
    else:
        max_acc = 0.0
    
    results = {
        "mean_error": mean_error,
        "max_error": max_error,
        "min_distance_overall": min_dist_overall,
        "collision_occurred": collision_occurred,
        "max_acceleration": max_acc,
    }
    
    if verbose:
        print("\n" + "="*50)
        print("VALIDATION RESULTS - SUB-PROBLEM 2")
        print("="*50)
        print(f"Final Formation Error:")
        print(f"  Mean error:     {mean_error:.4f}")
        print(f"  Max error:      {max_error:.4f}")
        print(f"\nCollision Safety:")
        print(f"  Min distance:   {min_dist_overall:.4f}")
        print(f"  Safety radius:  {rsafe:.4f}")
        print(f"  Status:         {'COLLISION' if collision_occurred else 'SAFE'}")
        print(f"\nTrajectory Smoothness:")
        print(f"  Max acceleration: {max_acc:.4f}")
        print("="*50 + "\n")
    
    return results


def plot_transition_metrics(traj: np.ndarray, T0: np.ndarray, T1: np.ndarray, tgrid: np.ndarray, T_ramp: float):
    """
    Plot metrics showing transition progress.
    
    Metrics:
      1. Distance from current formation to T1 (should decay)
      2. Alpha parameter (transition progress)
    """
    T = traj.shape[0]
    
    errors_to_T1 = np.zeros(T)
    alpha_vals = np.minimum(1.0, tgrid / T_ramp)
    
    for t in range(T):
        errors_to_T1[t] = np.mean(np.linalg.norm(traj[t] - T1, axis=1))
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Error to final target
    axes[0].plot(tgrid, errors_to_T1, linewidth=2, color='blue')
    axes[0].axvline(T_ramp, color='red', linestyle='--', label=f'T_ramp={T_ramp}s')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Mean distance to final target')
    axes[0].set_title('Transition Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Alpha parameter
    axes[1].plot(tgrid, alpha_vals, linewidth=2, color='green')
    axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('x(t) - interpolation parameter')
    axes[1].set_title('Target Interpolation: T(t) = (1-x)T0 + xT1')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.1, 1.1])
    
    plt.tight_layout()
    plt.savefig('transition_metrics_subproblem2.png', dpi=150)
    print("Saved: transition_metrics_subproblem2.png")
    plt.show()


# ========================================
# VISUALIZATION
# ========================================

def animate_trajectories(
    traj: np.ndarray,
    T0: np.ndarray,
    T1: np.ndarray,
    interval_ms: int = 25,
    save_mp4: bool = False,
):
    """
    Animated visualization of formation transition.
    
    Features:
      - Both target formations shown for reference
      - Drones transition from T0 to T1
      - Dark background with illumination effect
      - Clean visualization (no motion trails)
    """
    Tsteps, N, _ = traj.shape

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_aspect("equal", adjustable="box")
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Compute bounds
    xy = np.vstack([traj.reshape(-1, 2), T0, T1])
    pad = 0.5
    xmin, ymin = xy.min(axis=0) - pad
    xmax, ymax = xy.max(axis=0) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Sub-problem 2: Transition to 'Happy New Year!'", color='white', fontsize=14)
    ax.tick_params(colors='white')

    # Show target formation
    ax.scatter(T1[:, 0], T1[:, 1], s=6, c='magenta', alpha=0.15, marker='+', label='Final (greeting)')

    # Drones
    scat = ax.scatter(
        traj[0, :, 0], traj[0, :, 1],
        s=25, c='yellow', edgecolors='white',
        linewidths=0.5, alpha=0.9,
        label='Drones'
    )

    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

    time_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes,
        color='white', fontsize=10,
        verticalalignment='top'
    )

    def update(frame: int):
        pts = traj[frame]
        scat.set_offsets(pts)

        time_text.set_text(f'Frame: {frame + 1}/{Tsteps}')
        return (scat, time_text)

    anim = FuncAnimation(fig, update, frames=Tsteps, interval=interval_ms, blit=False)

    if save_mp4:
        print("Saving animation...")
        anim.save("subproblem2_transition.mp4", dpi=150, writer='ffmpeg')
        print("Saved: subproblem2_transition.mp4")

    plt.show()
    return anim  # Keep reference to prevent garbage collection


# ==============================
# MAIN EXECUTION
# ==============================


# Input paths
HANDWRITTEN_IMAGE = os.path.join("data", "handwritten_name.jpg")
TRAJ1_PATH = "trajectories_subproblem1.npy"

# Must match Sub-problem 1
N_DRONES = 300
SHAPE_SCALE = 1.0

# Physics parameters (same as Sub-problem 1)
params = dict(
    m=1.0,
    kp=20.0,
    kd=6.0,
    vmax=2.0,
    krep=0.05,
    rsafe=0.1,
)

# Time integration
dt = 0.01
t_end = 10.0
T_ramp = 5.0  # Transition duration (targets change over this time)

print("="*60)
print("DRONE SHOW - SUB-PROBLEM 2: TRANSITION")
print("="*60)
print(f"Configuration:")
print(f"  Handwritten image: {HANDWRITTEN_IMAGE}")
print(f"  Sub-1 trajectories: {TRAJ1_PATH}")
print(f"  Drones: {N_DRONES}")
print(f"  Transition time: {T_ramp}s")
print("="*60 + "\n")

# ============================================
# LOAD FINAL STATE FROM SUB-PROBLEM 1
# ============================================
print("Loading Sub-problem 1 results...")

if not os.path.exists(TRAJ1_PATH):
    raise FileNotFoundError(
        f"Cannot find {TRAJ1_PATH}. "
        f"Please run Sub-problem 1 first!"
    )

traj1 = np.load(TRAJ1_PATH)

if traj1.ndim != 3 or traj1.shape[1] != N_DRONES or traj1.shape[2] != 2:
    raise ValueError(
        f"Unexpected trajectory shape {traj1.shape}. "
        f"Expected (T, {N_DRONES}, 2). "
        f"Check N_DRONES matches Sub-problem 1."
    )

# Extract final positions and velocities
x_start = traj1[-1].copy()

# Approximate final velocity from last two positions
if traj1.shape[0] > 1:
    v_start = (traj1[-1] - traj1[-2]) / dt
else:
    v_start = np.zeros_like(x_start)

print(f"  OK Loaded final state from Sub-1: {traj1.shape[0]} timesteps")
print(f"  OK Starting from positions at t={traj1.shape[0]*dt:.2f}s")

# ============================================
# BUILD INITIAL FORMATION TARGETS (T0 = name)
# ============================================
print("\nExtracting handwritten name shape...")

pts_name, bw_name, skel_name = extract_shape_points_skeleton(
    HANDWRITTEN_IMAGE,
    n_points=N_DRONES,
    block_size=31,
    C=5,
    close_ksize=3,
    min_area=300,
    min_span=120,
    seed=0
)

_, r_name = shape_center_and_offsets(pts_name, scale_to=SHAPE_SCALE)
r_name[:, 1] *= -1

print(f"  OK Extracted {N_DRONES} points from name")

if DEBUG_PLOTS:
    plt.figure()
    plt.imshow(skel_name, cmap='gray')
    plt.title("Name Skeleton")
    plt.show()

# ============================================
# BUILD FINAL FORMATION TARGETS (T1 = greeting)
# ============================================
print("\nRendering 'Happy New Year!' greeting...")

greeting_path = render_text_image(
    "HAPPY NEW YEAR!",
    out_path="greeting.png",
    width=1600,
    height=500,
    font_scale=2.4,
    thickness=2
)

print("Extracting greeting shape...")

pts_greet, bw_greet, skel_greet = extract_shape_points_skeleton(
    greeting_path,
    n_points=N_DRONES,
    block_size=31,
    C=5,
    close_ksize=1,
    min_area=200,
    min_span=80,
    seed=1,
    min_dot_area=15
)

_, r_greet = shape_center_and_offsets(pts_greet, scale_to=SHAPE_SCALE)
r_greet[:, 1] *= -1

print(f"  OK Extracted {N_DRONES} points from greeting")

if DEBUG_PLOTS:
    plt.figure()
    plt.imshow(skel_greet, cmap='gray')
    plt.title("Greeting Skeleton")
    plt.show()

# ============================================
# RESCALE GREETING TO MATCH NAME SIZE
# ============================================
print("\nRescaling greeting to match name size...")

r_greet = rescale_to_match_bbox(r_greet, r_name, match="height")

print("  OK Shapes rescaled to similar size")

# ============================================
# ASSIGN GREETING TARGETS TO DRONES
# ============================================
print("\nAssigning greeting targets to drones (minimizing crossings)...")

T1_assigned = greedy_match_targets(x_start, r_greet)

print("  OK Target assignment complete")

# Optional: Plot both target formations
if DEBUG_PLOTS or True:
    plt.figure(figsize=(12, 5))
    plt.scatter(r_name[:, 0], r_name[:, 1], s=8, alpha=0.4, label="Name (T0)", c='blue')
    plt.scatter(T1_assigned[:, 0], T1_assigned[:, 1], s=8, alpha=0.4, label="Greeting (T1)", c='red')
    plt.axis("equal")
    plt.legend()
    plt.title("Sub-problem 2: Target Formations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("target_formations_subproblem2.png", dpi=150)
    print("Saved: target_formations_subproblem2.png")
    plt.show()

# ============================================
# SIMULATE TRANSITION
# ============================================
print("\nSimulating transition (this may take a minute)...")

traj2, tgrid2 = simulate_transition(
    x0=x_start,
    v0=v_start,
    T0=r_name,
    T1=T1_assigned,
    params=params,
    dt=dt,
    t_end=t_end,
    T_ramp=T_ramp,
)

print(f"  OK Simulation completed: {traj2.shape[0]} timesteps, final t={tgrid2[-1]:.2f}s")

# ============================================
# VALIDATE & SAVE
# ============================================
print("\nValidating solution...")

metrics = validate_transition(traj2, T1_assigned, params["rsafe"], verbose=True)

# Save trajectories
output_file = "trajectories_subproblem2.npy"
np.save(output_file, traj2)
print(f"Saved trajectories: {output_file}")

# Plot metrics
plot_transition_metrics(traj2, r_name, T1_assigned, tgrid2, T_ramp)

# ============================================
# VISUALIZE
# ============================================
print("\nCreating visualization...")

anim = animate_trajectories(
    traj2,
    r_name,
    T1_assigned,
    interval_ms=25,
    save_mp4=False
)

print("\n" + "="*60)
print("SUB-PROBLEM 2 COMPLETED SUCCESSFULLY!")
print("="*60)

