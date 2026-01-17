use visionctl::{Region, VisionCtl};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create headless controller
    let mut ctl = VisionCtl::new_headless();

    // 2. Define a region (e.g., 500x500 at 100,100)
    let region = Region::new(100, 100, 500, 500);
    println!("Setting viewport to {:?}", region);
    ctl.set_viewport(Some(region));

    // 3. Test conversion: Normalized Center (500, 500) -> Global Pixels
    // Expected: 100 + (500/2) = 350, 100 + (500/2) = 350
    let (px, py) = ctl.to_screen_coords(500, 500)?;
    println!("Normalized (500, 500) -> Screen ({}, {})", px, py);
    assert_eq!(px, 350);
    assert_eq!(py, 350);

    // 4. Test conversion: Normalized Top-Left (0, 0) -> Global Pixels (100, 100)
    let (px, py) = ctl.to_screen_coords(0, 0)?;
    println!("Normalized (0, 0) -> Screen ({}, {})", px, py);
    assert_eq!(px, 100);
    assert_eq!(py, 100);

    // 5. Test conversion: Global (350, 350) -> Normalized (500, 500)
    let norm = ctl.to_normalized_coords(350, 350)?;
    println!("Screen (350, 350) -> Normalized {:?}", norm);
    assert_eq!(norm, Some((500, 500)));

    // 6. Test conversion: Out of bounds Global (0, 0) -> None
    let norm = ctl.to_normalized_coords(0, 0)?;
    println!("Screen (0, 0) -> Normalized {:?}", norm);
    assert_eq!(norm, None);

    println!("All region tests passed!");
    Ok(())
}
