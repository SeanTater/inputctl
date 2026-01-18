//! Integration tests for inputctl
//!
//! These tests require access to /dev/uinput and are marked #[ignore].
//! Run with: sudo cargo test -- --ignored

use evdev::Key;
use inputctl::{Curve, InputCtl, MouseButton};

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn create_device() {
    let yd = InputCtl::new();
    assert!(yd.is_ok(), "should create device: {:?}", yd.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn type_text() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.type_text("hello");
    assert!(result.is_ok(), "should type text: {:?}", result.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn mouse_click() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.click(MouseButton::Left);
    assert!(result.is_ok(), "should click: {:?}", result.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn mouse_move() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.move_mouse(10, 20);
    assert!(result.is_ok(), "should move mouse: {:?}", result.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn scroll() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.scroll(3);
    assert!(result.is_ok(), "should scroll: {:?}", result.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn mouse_move_smooth_linear() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.move_mouse_smooth(100, 50, 0.5, Curve::Linear, 2.0, 60);
    assert!(
        result.is_ok(),
        "should move mouse smoothly: {:?}",
        result.err()
    );
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn mouse_move_smooth_ease_in_out() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.move_mouse_smooth(100, 50, 1.0, Curve::EaseInOut, 2.0, 60);
    assert!(
        result.is_ok(),
        "should move mouse smoothly with ease-in-out: {:?}",
        result.err()
    );
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn mouse_move_smooth_negative() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.move_mouse_smooth(-50, -30, 0.3, Curve::Linear, 2.0, 60);
    assert!(
        result.is_ok(),
        "should move mouse smoothly in negative direction: {:?}",
        result.err()
    );
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn mouse_move_smooth_no_noise() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.move_mouse_smooth(100, 50, 1.0, Curve::Linear, 0.0, 60);
    assert!(
        result.is_ok(),
        "should move mouse smoothly with no noise: {:?}",
        result.err()
    );
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_hold_state_tracking() {
    let mut ctl = InputCtl::new().expect("failed to create device");
    assert!(
        !ctl.is_key_held(Key::KEY_A),
        "key should not be held initially"
    );

    ctl.key_down(Key::KEY_A).expect("should press key down");
    assert!(
        ctl.is_key_held(Key::KEY_A),
        "key should be held after key_down"
    );

    ctl.key_up(Key::KEY_A).expect("should release key");
    assert!(
        !ctl.is_key_held(Key::KEY_A),
        "key should not be held after key_up"
    );
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_mouse_button_hold_state() {
    let mut ctl = InputCtl::new().expect("failed to create device");
    assert!(
        !ctl.is_mouse_button_held(MouseButton::Left),
        "button should not be held initially"
    );

    ctl.mouse_down(MouseButton::Left)
        .expect("should press button down");
    assert!(
        ctl.is_mouse_button_held(MouseButton::Left),
        "button should be held after mouse_down"
    );

    ctl.mouse_up(MouseButton::Left)
        .expect("should release button");
    assert!(
        !ctl.is_mouse_button_held(MouseButton::Left),
        "button should not be held after mouse_up"
    );
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_release_all() {
    let mut ctl = InputCtl::new().expect("failed to create device");

    // Hold multiple keys and buttons
    ctl.key_down(Key::KEY_LEFTSHIFT)
        .expect("should press shift");
    ctl.key_down(Key::KEY_LEFTCTRL).expect("should press ctrl");
    ctl.mouse_down(MouseButton::Left)
        .expect("should press left button");

    assert_eq!(ctl.get_held_keys().len(), 2, "should have 2 held keys");
    assert_eq!(ctl.get_held_buttons().len(), 1, "should have 1 held button");

    // Release all
    ctl.release_all().expect("should release all");

    assert!(
        !ctl.is_key_held(Key::KEY_LEFTSHIFT),
        "shift should not be held"
    );
    assert!(
        !ctl.is_key_held(Key::KEY_LEFTCTRL),
        "ctrl should not be held"
    );
    assert!(
        !ctl.is_mouse_button_held(MouseButton::Left),
        "left button should not be held"
    );
    assert_eq!(ctl.get_held_keys().len(), 0, "should have no held keys");
    assert_eq!(
        ctl.get_held_buttons().len(),
        0,
        "should have no held buttons"
    );
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_idempotent_key_down() {
    let mut ctl = InputCtl::new().expect("failed to create device");

    // Press the same key twice
    ctl.key_down(Key::KEY_A)
        .expect("first key_down should succeed");
    ctl.key_down(Key::KEY_A)
        .expect("second key_down should not error");

    assert_eq!(ctl.get_held_keys().len(), 1, "should only track key once");
    assert!(ctl.is_key_held(Key::KEY_A), "key should still be held");
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_idempotent_key_up() {
    let mut ctl = InputCtl::new().expect("failed to create device");

    ctl.key_down(Key::KEY_A).expect("should press key");
    ctl.key_up(Key::KEY_A).expect("first key_up should succeed");
    ctl.key_up(Key::KEY_A)
        .expect("second key_up should not error");

    assert!(!ctl.is_key_held(Key::KEY_A), "key should not be held");
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_get_held_keys() {
    let mut ctl = InputCtl::new().expect("failed to create device");

    assert!(
        ctl.get_held_keys().is_empty(),
        "should start with no held keys"
    );

    ctl.key_down(Key::KEY_A).expect("should press A");
    ctl.key_down(Key::KEY_B).expect("should press B");
    ctl.key_down(Key::KEY_LEFTSHIFT)
        .expect("should press shift");

    let held = ctl.get_held_keys();
    assert_eq!(held.len(), 3, "should have 3 held keys");
    assert!(held.contains(&Key::KEY_A), "should contain KEY_A");
    assert!(held.contains(&Key::KEY_B), "should contain KEY_B");
    assert!(
        held.contains(&Key::KEY_LEFTSHIFT),
        "should contain KEY_LEFTSHIFT"
    );
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_get_held_buttons() {
    let mut ctl = InputCtl::new().expect("failed to create device");

    assert!(
        ctl.get_held_buttons().is_empty(),
        "should start with no held buttons"
    );

    ctl.mouse_down(MouseButton::Left)
        .expect("should press left");
    ctl.mouse_down(MouseButton::Right)
        .expect("should press right");

    let held = ctl.get_held_buttons();
    assert_eq!(held.len(), 2, "should have 2 held buttons");
    assert!(held.contains(&MouseButton::Left), "should contain Left");
    assert!(held.contains(&MouseButton::Right), "should contain Right");
}

#[test]
#[ignore = "requires /dev/uinput access and KDE Plasma (run with sudo)"]
fn test_mouse_movement_accuracy() {
    use inputctl_capture::find_cursor;

    println!("\n=== Mouse Movement Accuracy Test ===\n");

    // Get initial cursor position
    let pos_before = find_cursor().expect("Failed to get initial cursor position");
    println!(
        "Initial cursor position: ({}, {})",
        pos_before.x, pos_before.y
    );

    // Create inputctl device
    let mut ctl = InputCtl::new().expect("Failed to create input device");

    // Test cases: (dx, dy, description)
    let test_cases = vec![
        (100, 0, "right 100px"),
        (0, 100, "down 100px"),
        (-100, 0, "left 100px"),
        (0, -100, "up 100px"),
        (50, 50, "diagonal 50,50"),
        (-50, -50, "diagonal -50,-50"),
    ];

    let mut failures = Vec::new();
    let mut scale_factors = Vec::new();

    for (dx, dy, desc) in test_cases {
        println!("\nTest: Move {} ({}, {})", desc, dx, dy);

        // Get position before move
        let before = find_cursor().expect("Failed to get cursor position before move");
        println!("  Before: ({}, {})", before.x, before.y);

        // Move mouse
        ctl.move_mouse(dx, dy).expect("Failed to move mouse");

        // Wait for movement to complete
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Get position after move
        let after = find_cursor().expect("Failed to get cursor position after move");
        println!("  After:  ({}, {})", after.x, after.y);

        // Calculate actual movement
        let actual_dx = after.x - before.x;
        let actual_dy = after.y - before.y;
        println!("  Requested: ({}, {})", dx, dy);
        println!("  Actual:    ({}, {})", actual_dx, actual_dy);

        // Calculate error
        let error_x = (actual_dx - dx).abs();
        let error_y = (actual_dy - dy).abs();
        println!("  Error:     ({}, {})", error_x, error_y);

        // Calculate scaling factors
        let scale_x = if dx != 0 {
            actual_dx as f64 / dx as f64
        } else {
            1.0
        };
        let scale_y = if dy != 0 {
            actual_dy as f64 / dy as f64
        } else {
            1.0
        };
        if dx != 0 || dy != 0 {
            println!("  Scale:     ({:.4}, {:.4})", scale_x, scale_y);
            if dx != 0 {
                scale_factors.push(scale_x);
            }
            if dy != 0 {
                scale_factors.push(scale_y);
            }
        }

        // Check if error is within tolerance
        let max_error = 2;
        if error_x > max_error || error_y > max_error {
            failures.push(format!(
                "{}: expected ({}, {}), got ({}, {}), error ({}, {})",
                desc, dx, dy, actual_dx, actual_dy, error_x, error_y
            ));
        }
    }

    // Calculate average scaling factor
    if !scale_factors.is_empty() {
        let avg_scale: f64 = scale_factors.iter().sum::<f64>() / scale_factors.len() as f64;
        println!("\n=== Scaling Analysis ===");
        println!("Average scaling factor: {:.4}", avg_scale);
        println!("Compensation factor needed: {:.4}", 1.0 / avg_scale);
        println!("All scale factors: {:?}", scale_factors);
    }

    if !failures.is_empty() {
        println!("\n=== FAILURES ===");
        for failure in &failures {
            println!("  {}", failure);
        }
        panic!(
            "\n{} test cases failed. See scaling analysis above.",
            failures.len()
        );
    }

    println!("\n=== All movement tests passed! ===\n");
}

#[test]
#[ignore = "requires sudo, KDE Plasma, and single-monitor (random positions may be unreachable on multi-monitor)"]
fn test_smooth_mouse_movement_accuracy() {
    use inputctl_capture::find_cursor;

    println!("\n=== Smooth Mouse Movement Accuracy Test ===\n");

    // Get initial cursor position
    let pos_before = find_cursor().expect("Failed to get initial cursor position");
    println!(
        "Initial cursor position: ({}, {})",
        pos_before.x, pos_before.y
    );

    // Create inputctl device
    let mut ctl = InputCtl::new().expect("Failed to create input device");

    // Test cases: (dx, dy, duration, curve, noise, description)
    let test_cases = vec![
        (200, 0, 0.5, Curve::Linear, 0.0, "right 200px (no noise)"),
        (
            0,
            200,
            0.5,
            Curve::EaseInOut,
            0.0,
            "down 200px (no noise, ease)",
        ),
        (
            -200,
            0,
            0.5,
            Curve::Linear,
            2.0,
            "left 200px (default noise)",
        ),
        (
            0,
            -200,
            0.5,
            Curve::EaseInOut,
            2.0,
            "up 200px (default noise, ease)",
        ),
    ];

    let mut failures = Vec::new();
    let mut errors_x = Vec::new();
    let mut errors_y = Vec::new();

    for (dx, dy, duration, curve, noise, desc) in test_cases {
        println!("\nTest: Move {} (noise={})", desc, noise);

        // Get position before move
        let before = find_cursor().expect("Failed to get cursor position before move");
        println!("  Before: ({}, {})", before.x, before.y);

        // Move mouse smoothly
        ctl.move_mouse_smooth(dx, dy, duration, curve, noise, 60)
            .expect("Failed to move mouse smoothly");

        // Wait for movement to complete
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Get position after move
        let after = find_cursor().expect("Failed to get cursor position after move");
        println!("  After:  ({}, {})", after.x, after.y);

        // Calculate actual movement
        let actual_dx = after.x - before.x;
        let actual_dy = after.y - before.y;
        println!("  Requested: ({}, {})", dx, dy);
        println!("  Actual:    ({}, {})", actual_dx, actual_dy);

        // Calculate error
        let error_x = actual_dx - dx; // Keep sign to see if it's over/under
        let error_y = actual_dy - dy;
        println!("  Error:     ({}, {}) [+/-]", error_x, error_y);

        if dx != 0 {
            errors_x.push(error_x);
        }
        if dy != 0 {
            errors_y.push(error_y);
        }

        // Check if error is within tolerance
        let max_error = 3;
        if error_x.abs() > max_error || error_y.abs() > max_error {
            failures.push(format!(
                "{}: expected ({}, {}), got ({}, {}), error ({}, {})",
                desc, dx, dy, actual_dx, actual_dy, error_x, error_y
            ));
        }
    }

    // Test random movements to check consistency
    println!("\n=== Random Movement Tests ===");
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for i in 0..20 {
        let dx = rng.gen_range(-300..300);
        let dy = rng.gen_range(-300..300);
        let duration = 0.3;

        let before = find_cursor().expect("Failed to get cursor position");
        ctl.move_mouse_smooth(dx, dy, duration, Curve::Linear, 0.0, 60)
            .expect("Failed to move mouse smoothly");
        std::thread::sleep(std::time::Duration::from_millis(100));
        let after = find_cursor().expect("Failed to get cursor position");

        let actual_dx = after.x - before.x;
        let actual_dy = after.y - before.y;
        let error_x = actual_dx - dx;
        let error_y = actual_dy - dy;

        println!(
            "Random #{}: req=({}, {}), actual=({}, {}), error=({}, {})",
            i + 1,
            dx,
            dy,
            actual_dx,
            actual_dy,
            error_x,
            error_y
        );

        if dx != 0 {
            errors_x.push(error_x);
        }
        if dy != 0 {
            errors_y.push(error_y);
        }
    }

    // Analyze error patterns
    println!("\n=== Error Analysis ===");
    if !errors_x.is_empty() {
        let avg_x: f64 = errors_x.iter().map(|&e| e as f64).sum::<f64>() / errors_x.len() as f64;
        let min_x = errors_x.iter().min().unwrap();
        let max_x = errors_x.iter().max().unwrap();
        println!(
            "X-axis errors: min={}, max={}, avg={:.2}",
            min_x, max_x, avg_x
        );
        println!("All X errors: {:?}", errors_x);
    }
    if !errors_y.is_empty() {
        let avg_y: f64 = errors_y.iter().map(|&e| e as f64).sum::<f64>() / errors_y.len() as f64;
        let min_y = errors_y.iter().min().unwrap();
        let max_y = errors_y.iter().max().unwrap();
        println!(
            "Y-axis errors: min={}, max={}, avg={:.2}",
            min_y, max_y, avg_y
        );
        println!("All Y errors: {:?}", errors_y);
    }

    if !failures.is_empty() {
        println!("\n=== FAILURES ===");
        for failure in &failures {
            println!("  {}", failure);
        }
        panic!(
            "\n{} test cases failed. See error analysis above.",
            failures.len()
        );
    }

    println!("\n=== All smooth movement tests passed! ===\n");
}
