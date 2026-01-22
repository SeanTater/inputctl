use inputctl_capture::recorder_ops::pipewire_to_ffmpeg_format;

#[test]
fn pipewire_format_maps_known() {
    let (fmt, unknown) = pipewire_to_ffmpeg_format("BGRA");
    assert_eq!(fmt, "bgra");
    assert!(unknown.is_none());

    let (fallback, unknown) = pipewire_to_ffmpeg_format("weird");
    assert_eq!(fallback, "bgr0");
    assert_eq!(unknown, Some("weird"));
}
