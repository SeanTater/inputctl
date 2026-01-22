use inputctl_capture::recorder_ops::build_ffmpeg_args;
use inputctl_capture::Encoder;

#[test]
fn build_ffmpeg_args_includes_preset() {
    let args = build_ffmpeg_args(
        &Encoder::X264,
        None,
        640,
        480,
        10,
        28,
        "bgr0",
        "veryfast",
        &std::path::PathBuf::from("/tmp/out.mp4"),
        None,
    );

    let preset_pos = args.iter().position(|arg| arg == "-preset").unwrap();
    assert_eq!(args[preset_pos + 1], "veryfast");
}
