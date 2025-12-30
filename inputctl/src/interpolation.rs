use rand::Rng;

/// Interpolation curve types for smooth mouse movement
#[derive(Debug, Clone, Copy)]
pub enum Curve {
    /// Linear interpolation (constant velocity)
    Linear,
    /// Cubic ease-in-out (smooth acceleration and deceleration)
    EaseInOut,
}

/// Calculate interpolated value based on curve type
///
/// # Arguments
/// * `t` - Progress from 0.0 to 1.0
/// * `curve` - The interpolation curve to use
///
/// # Returns
/// Interpolated value from 0.0 to 1.0
fn interpolate(t: f64, curve: &Curve) -> f64 {
    match curve {
        Curve::Linear => t,
        Curve::EaseInOut => {
            // Cubic ease-in-out: smooth start and end
            if t < 0.5 {
                4.0 * t * t * t
            } else {
                1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
            }
        }
    }
}

/// Smooth cubic interpolation (smoothstep)
///
/// # Arguments
/// * `t` - Value from 0.0 to 1.0
///
/// # Returns
/// Smoothly interpolated value using cubic Hermite curve (3t² - 2t³)
fn smoothstep(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Generate smooth low-frequency noise at position t
///
/// Uses cubic interpolation between random control points to create
/// organic, natural-looking variation.
///
/// # Arguments
/// * `t` - Position along path from 0.0 to 1.0
/// * `control_points` - Array of (position, value) tuples
///
/// # Returns
/// Smooth noise value interpolated from control points
fn smooth_noise(t: f64, control_points: &[(f64, f64)]) -> f64 {
    if control_points.len() < 2 {
        return 0.0;
    }

    // Find surrounding control points
    let mut idx = 0;
    for (i, &(pos, _)) in control_points.iter().enumerate() {
        if pos > t {
            break;
        }
        idx = i;
    }

    // Handle edge cases
    if idx >= control_points.len() - 1 {
        return control_points.last().unwrap().1;
    }

    let (t0, v0) = control_points[idx];
    let (t1, v1) = control_points[idx + 1];

    // Interpolate between control points using smoothstep
    let local_t = (t - t0) / (t1 - t0);
    let smooth_t = smoothstep(local_t);

    v0 + (v1 - v0) * smooth_t
}

/// Generate movement steps with interpolation and smooth noise
///
/// Creates a series of small movements that smoothly transition from
/// start to end using the specified curve, with low-frequency smooth noise
/// for natural-looking variation.
///
/// # Arguments
/// * `dx` - Total horizontal movement (positive = right)
/// * `dy` - Total vertical movement (positive = down)
/// * `duration` - Time in seconds for the movement
/// * `curve` - Interpolation curve type
/// * `noise_amount` - Maximum deviation in pixels (e.g., 2.0 = ±2 pixels). Use 0.0 for no noise.
///
/// # Returns
/// Vector of (dx, dy) movement deltas
pub fn generate_steps(dx: i32, dy: i32, duration: f64, curve: Curve, noise_amount: f64) -> Vec<(i32, i32)> {
    let mut rng = rand::thread_rng();

    // Target 125 Hz (8ms intervals) for smooth movement
    // Minimum 10 steps to ensure smoothness even for short durations
    let steps = (duration * 125.0).ceil().max(10.0) as usize;
    let mut movements = Vec::with_capacity(steps);

    // Generate control points for smooth low-frequency noise
    // Control point every ~25 steps (200ms) for natural variation
    let control_interval = 25;
    let num_controls = (steps / control_interval).max(2) + 2;
    let mut control_points_x = Vec::with_capacity(num_controls);
    let mut control_points_y = Vec::with_capacity(num_controls);

    // First and last control points at 0 for smooth start/end
    for i in 0..num_controls {
        let t = i as f64 / (num_controls - 1) as f64;
        // Apply configurable noise amount
        let noise_x = if i == 0 || i == num_controls - 1 {
            0.0 // Start and end at target
        } else {
            rng.gen_range(-noise_amount..=noise_amount)
        };
        let noise_y = if i == 0 || i == num_controls - 1 {
            0.0 // Start and end at target
        } else {
            rng.gen_range(-noise_amount..=noise_amount)
        };
        control_points_x.push((t, noise_x));
        control_points_y.push((t, noise_y));
    }

    let mut emitted_x = 0;
    let mut emitted_y = 0;

    for i in 0..steps {
        let t = (i + 1) as f64 / steps as f64;
        let progress = interpolate(t, &curve);

        // Apply smooth low-frequency noise
        let noise_x = smooth_noise(t, &control_points_x);
        let noise_y = smooth_noise(t, &control_points_y);

        // Calculate target position with smooth noise
        let target_x = (dx as f64 * progress + noise_x).round() as i32;
        let target_y = (dy as f64 * progress + noise_y).round() as i32;

        // Calculate delta from last emitted position
        let step_x = target_x - emitted_x;
        let step_y = target_y - emitted_y;

        emitted_x = target_x;
        emitted_y = target_y;

        movements.push((step_x, step_y));
    }

    // Final correction to ensure we hit target exactly
    let correction_x = dx - emitted_x;
    let correction_y = dy - emitted_y;
    if let Some(last) = movements.last_mut() {
        last.0 += correction_x;
        last.1 += correction_y;
    }

    movements
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_interpolation_boundaries() {
        assert_eq!(interpolate(0.0, &Curve::Linear), 0.0);
        assert_eq!(interpolate(0.5, &Curve::Linear), 0.5);
        assert_eq!(interpolate(1.0, &Curve::Linear), 1.0);
    }

    #[test]
    fn ease_in_out_boundaries() {
        let result_start = interpolate(0.0, &Curve::EaseInOut);
        let result_mid = interpolate(0.5, &Curve::EaseInOut);
        let result_end = interpolate(1.0, &Curve::EaseInOut);

        assert!((result_start - 0.0).abs() < 1e-10);
        assert!((result_mid - 0.5).abs() < 1e-10);
        assert!((result_end - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ease_in_out_is_smooth() {
        // Ease-in-out should be symmetric
        let t1 = interpolate(0.25, &Curve::EaseInOut);
        let t2 = interpolate(0.75, &Curve::EaseInOut);
        assert!((t1 + t2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn generate_steps_reaches_target() {
        let steps = generate_steps(100, 50, 0.5, Curve::Linear, 2.0);

        // Sum all steps to verify we reach the exact target
        let total_x: i32 = steps.iter().map(|(x, _)| x).sum();
        let total_y: i32 = steps.iter().map(|(_, y)| y).sum();

        assert_eq!(total_x, 100);
        assert_eq!(total_y, 50);
    }

    #[test]
    fn generate_steps_minimum_count() {
        // Even short durations should have minimum steps
        let steps = generate_steps(10, 5, 0.01, Curve::Linear, 2.0);
        assert!(steps.len() >= 10);
    }

    #[test]
    fn generate_steps_scales_with_duration() {
        let steps_short = generate_steps(100, 50, 0.5, Curve::Linear, 2.0);
        let steps_long = generate_steps(100, 50, 1.0, Curve::Linear, 2.0);

        // Longer duration should produce more steps
        assert!(steps_long.len() > steps_short.len());

        // Should be approximately 125 Hz
        assert!((steps_long.len() as f64 - 125.0).abs() < 5.0);
    }

    #[test]
    fn generate_steps_negative_movement() {
        let steps = generate_steps(-100, -50, 0.5, Curve::Linear, 2.0);

        let total_x: i32 = steps.iter().map(|(x, _)| x).sum();
        let total_y: i32 = steps.iter().map(|(_, y)| y).sum();

        assert_eq!(total_x, -100);
        assert_eq!(total_y, -50);
    }

    #[test]
    fn generate_steps_ease_in_out_reaches_target() {
        let steps = generate_steps(200, 100, 1.0, Curve::EaseInOut, 2.0);

        let total_x: i32 = steps.iter().map(|(x, _)| x).sum();
        let total_y: i32 = steps.iter().map(|(_, y)| y).sum();

        assert_eq!(total_x, 200);
        assert_eq!(total_y, 100);
    }

    #[test]
    fn generate_steps_no_noise() {
        let steps = generate_steps(100, 50, 1.0, Curve::Linear, 0.0);

        // With no noise, should still reach target
        let total_x: i32 = steps.iter().map(|(x, _)| x).sum();
        let total_y: i32 = steps.iter().map(|(_, y)| y).sum();

        assert_eq!(total_x, 100);
        assert_eq!(total_y, 50);
    }

    #[test]
    fn smoothstep_boundaries() {
        assert_eq!(smoothstep(0.0), 0.0);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-10);
        assert_eq!(smoothstep(1.0), 1.0);
    }

    #[test]
    fn smooth_noise_interpolates() {
        let control_points = vec![(0.0, 0.0), (1.0, 10.0)];

        let v0 = smooth_noise(0.0, &control_points);
        let v_mid = smooth_noise(0.5, &control_points);
        let v1 = smooth_noise(1.0, &control_points);

        assert!((v0 - 0.0).abs() < 1e-10);
        assert!(v_mid > 0.0 && v_mid < 10.0);
        assert!((v1 - 10.0).abs() < 1e-10);
    }
}
