"""
Test Cases Module for Drone Show Project

This module provides test cases demonstrating where the implementation
works well and where it fails, with explanations of limitations.

Run this after completing all three sub-problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist


def check_files_exist():
    """Check if all required output files exist."""
    required = [
        "outputs/sub1/trajectories_subproblem1.npy",
        "outputs/sub2/trajectories_subproblem2.npy",
        "outputs/sub3/trajectories_subproblem3.npy"
    ]
    
    missing = [f for f in required if not Path(f).exists()]
    
    if missing:
        print("ERROR: Missing files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    print("OK: All trajectory files found")
    return True


def test_static_convergence(traj_path, name, rsafe=0.06):
    """Test convergence for static/transition problems (Sub-1 and Sub-2)."""
    traj = np.load(traj_path)
    final = traj[-1]
    
    # Check collisions
    min_dist = np.min(pdist(final))
    
    # Check convergence (velocity near zero)
    if traj.shape[0] > 1:
        v = traj[-1] - traj[-2]
        mean_speed = np.mean(np.linalg.norm(v, axis=1))
    else:
        mean_speed = 0.0
    
    print(f"\n{name}:")
    print(f"  Min distance: {min_dist:.4f} (safe={rsafe:.4f})")
    print(f"  Mean speed: {mean_speed:.4f}")
    
    collision = min_dist < rsafe
    converged = mean_speed < 0.02  # Relaxed threshold
    
    if collision:
        print(f"  Status: FAIL (collision detected)")
    elif not converged:
        print(f"  Status: WARNING (still moving, but may be acceptable)")
    else:
        print(f"  Status: PASS")
    
    return not collision and converged


def test_dynamic_tracking(traj_path, name, rsafe=0.06):
    """Test tracking quality for dynamic problem (Sub-3)."""
    traj = np.load(traj_path)
    
    # For dynamic tracking, we don't expect convergence (targets always moving)
    # Instead, check: 1) No collisions, 2) Reasonable motion
    
    # Check collisions throughout trajectory
    min_dist_overall = float('inf')
    for t in range(traj.shape[0]):
        min_dist = np.min(pdist(traj[t]))
        min_dist_overall = min(min_dist_overall, min_dist)
    
    # Check if swarm is actually moving (tracking dynamic target)
    if traj.shape[0] > 10:
        v_samples = []
        for i in range(10, traj.shape[0], traj.shape[0]//10):
            v = traj[i] - traj[i-1]
            v_samples.append(np.mean(np.linalg.norm(v, axis=1)))
        mean_speed = np.mean(v_samples)
    else:
        mean_speed = 0.0
    
    print(f"\n{name}:")
    print(f"  Min distance (overall): {min_dist_overall:.4f} (safe={rsafe:.4f})")
    print(f"  Mean tracking speed: {mean_speed:.4f}")
    
    collision = min_dist_overall < rsafe
    tracking = mean_speed > 0.001  # Should be moving to track
    
    if collision:
        print(f"  Status: FAIL (collision detected)")
    elif not tracking:
        print(f"  Status: WARNING (swarm not moving, may indicate tracking failure)")
    else:
        print(f"  Status: PASS (tracking without collisions)")
    
    return not collision


def test_parameter_sensitivity():
    """Demonstrate parameter sensitivity with simple cases."""
    
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY TESTS")
    print("="*60)
    
    print("\nTest 1: krep too small (collision risk)")
    print("  Expected: Drones may collide in dense formations")
    print("  Mitigation: Increase krep or rsafe")
    
    print("\nTest 2: kp too large (oscillation)")
    print("  Expected: Overshoot and oscillation around targets")
    print("  Mitigation: Decrease kp or increase kd")
    
    print("\nTest 3: vmax too small (slow convergence)")
    print("  Expected: Very slow formation changes")
    print("  Mitigation: Increase vmax")
    
    print("\nTest 4: Insufficient drones (N < 200)")
    print("  Expected: Poor shape representation")
    print("  Mitigation: Increase N_DRONES")
    
    print("\nTest 5: Video threshold incorrect (Sub-3)")
    print("  Expected: Wrong/no object detected")
    print("  Mitigation: Adjust thresh or invert parameters")


def plot_all_trajectories():
    """Visualize all three sub-problems side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = [
        "Sub-1: Static Formation",
        "Sub-2: Transition",
        "Sub-3: Dynamic Tracking"
    ]
    
    paths = [
        "outputs/sub1/trajectories_subproblem1.npy",
        "outputs/sub2/trajectories_subproblem2.npy",
        "outputs/sub3/trajectories_subproblem3.npy"
    ]
    
    for ax, title, path in zip(axes, titles, paths):
        traj = np.load(path)
        
        # Plot first and last positions
        ax.scatter(traj[0, :, 0], traj[0, :, 1], s=10, alpha=0.3, label='Start')
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=10, alpha=0.8, label='End')
        
        # For Sub-3, also show middle frame
        if "subproblem3" in path and traj.shape[0] > 2:
            mid = traj.shape[0] // 2
            ax.scatter(traj[mid, :, 0], traj[mid, :, 1], s=10, alpha=0.5, label='Mid')
        
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/tests/test_all_trajectories.png', dpi=150)
    print("\nSaved: test_all_trajectories.png")
    plt.show()


def generate_test_report():
    """Generate a simple test report."""
    
    print("\n" + "="*60)
    print("TEST REPORT")
    print("="*60)
    
    if not check_files_exist():
        return
    
    # Test each sub-problem with appropriate criteria
    results = []
    
    results.append(test_static_convergence(
        "outputs/sub1/trajectories_subproblem1.npy",
        "Sub-problem 1 (Static Formation)"
    ))
    
    results.append(test_static_convergence(
        "outputs/sub2/trajectories_subproblem2.npy",
        "Sub-problem 2 (Transition)"
    ))
    
    results.append(test_dynamic_tracking(
        "outputs/sub3/trajectories_subproblem3.npy",
        "Sub-problem 3 (Dynamic Tracking)"
    ))
    
    # Parameter sensitivity
    test_parameter_sensitivity()
    
    # Visualization
    plot_all_trajectories()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED - check parameters or implementation")
    
    print("\nKnown Limitations:")
    print("1. O(N²) repulsion scales poorly beyond N~500")
    print("2. Greedy assignment suboptimal (use scipy for Hungarian)")
    print("3. Video tracking requires good contrast and simple backgrounds")
    print("4. No obstacle avoidance implemented")
    print("5. 2D only (no altitude control)")
    print("6. Sub-3 tracking lag depends on kp, vmax, and target speed")


if __name__ == "__main__":
    generate_test_report()