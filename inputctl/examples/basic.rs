use inputctl::{InputCtl, MouseButton};

fn main() -> inputctl::Result<()> {
    // Create device (takes ~1 second for kernel initialization)
    println!("Creating virtual input device...");
    let mut yd = InputCtl::new()?;
    println!("Device ready!");

    // Type some text
    yd.type_text("Hello, World!\n")?;

    // Click the mouse
    yd.click(MouseButton::Left)?;

    // Move the mouse
    yd.move_mouse(100, 50)?;

    // Scroll
    yd.scroll(3)?;

    Ok(())
}
