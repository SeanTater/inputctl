use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use image::{imageops::FilterType, DynamicImage};
use inputctl::{InputCtl, Key};
use inputctl_capture::{capture_screenshot_image, find_window, ScreenshotOptions};
use ndarray::{Array2, Array4};
use ort::{init, session::Session, value::Tensor};
use serde::Deserialize;
use std::collections::VecDeque;
use std::fs;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(author, version, about = "Reflex Agent ONNX inference runner")]
struct Cli {
    #[arg(long)]
    model: String,
    #[arg(long)]
    manifest: Option<String>,
    #[arg(long, default_value_t = 360)]
    height: usize,
    #[arg(long, default_value_t = 640)]
    width: usize,
    #[arg(long)]
    goal_intent: Option<String>,
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,
    #[arg(long, default_value_t = 2)]
    debounce: usize,
    #[arg(long, default_value_t = 4)]
    max_keys: usize,
    /// Use model's predicted intent instead of fixed goal (closed-loop)
    #[arg(long, default_value_t = false)]
    auto_intent: bool,
    /// Frames of consistent intent prediction before switching (prevents jitter)
    #[arg(long, default_value_t = 5)]
    intent_stability: usize,
    /// Print value estimates and intent predictions
    #[arg(long, short, default_value_t = false)]
    verbose: bool,
    #[command(subcommand)]
    command: CommandMode,
}

#[derive(Subcommand)]
enum CommandMode {
    Live {
        #[arg(long)]
        window: Option<String>,
        #[arg(long, default_value_t = 10.0)]
        fps: f32,
    },
}

#[derive(Deserialize)]
struct Manifest {
    inputs: ManifestInputs,
    outputs: ManifestOutputs,
}

#[derive(Deserialize)]
struct ManifestInputs {
    goal: ManifestGoal,
}

#[derive(Deserialize)]
struct ManifestGoal {
    intents: Vec<String>,
}

#[derive(Deserialize)]
struct ManifestOutputs {
    keys_logits: ManifestKeys,
}

#[derive(Deserialize)]
struct ManifestKeys {
    keys: Vec<String>,
}

struct InferenceEngine {
    session: Session,
    manifest: Manifest,
}

impl InferenceEngine {
    fn load(model_path: &str, manifest_path: &str) -> Result<Self> {
        let raw = fs::read_to_string(manifest_path)?;
        let manifest: Manifest = serde_json::from_str(&raw)?;
        init().with_name("reflex_infer").commit();
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self { session, manifest })
    }

    fn intent_index(&self, name: &str) -> Option<usize> {
        self.manifest
            .inputs
            .goal
            .intents
            .iter()
            .position(|v| v == name)
    }

    fn one_hot_goal(&self, idx: usize) -> Array2<f32> {
        let intents = self.manifest.inputs.goal.intents.len();
        let mut arr = Array2::<f32>::zeros((1, intents));
        if idx < intents {
            arr[[0, idx]] = 1.0;
        }
        arr
    }

    fn run(&mut self, pixels: &Array4<f32>, goal: &Array2<f32>) -> Result<(Vec<f32>, Vec<f32>, f32)> {
        let outputs = self.session.run(ort::inputs![
            "pixels" => Tensor::from_array(pixels.clone())?,
            "goal" => Tensor::from_array(goal.clone())?,
        ])?;

        let keys_logits = outputs[0].try_extract_array::<f32>()?;
        let intent_logits = outputs[2].try_extract_array::<f32>()?;
        let value_output = outputs[3].try_extract_array::<f32>()?;

        let keys = keys_logits.into_dimensionality::<ndarray::Ix2>()?;
        let intents = intent_logits.into_dimensionality::<ndarray::Ix2>()?;
        let value = value_output.into_dimensionality::<ndarray::Ix1>()?;

        Ok((keys.row(0).to_vec(), intents.row(0).to_vec(), value[0]))
    }

    fn _num_intents(&self) -> usize {
        self.manifest.inputs.goal.intents.len()
    }

    fn intent_name(&self, idx: usize) -> Option<&str> {
        self.manifest.inputs.goal.intents.get(idx).map(|s| s.as_str())
    }
}

/// Tracks intent prediction with hysteresis to prevent jitter
struct IntentState {
    current_intent: usize,
    candidate_intent: usize,
    candidate_count: usize,
    stability_threshold: usize,
}

impl IntentState {
    fn new(initial_intent: usize, stability_threshold: usize) -> Self {
        Self {
            current_intent: initial_intent,
            candidate_intent: initial_intent,
            candidate_count: 0,
            stability_threshold,
        }
    }

    /// Update with a new predicted intent, returns the (possibly unchanged) current intent
    fn update(&mut self, predicted_intent: usize) -> usize {
        if predicted_intent == self.current_intent {
            // Already at this intent, reset candidate tracking
            self.candidate_intent = predicted_intent;
            self.candidate_count = 0;
        } else if predicted_intent == self.candidate_intent {
            // Same candidate as before, increment count
            self.candidate_count += 1;
            if self.candidate_count >= self.stability_threshold {
                // Stable enough, switch
                self.current_intent = predicted_intent;
                self.candidate_count = 0;
            }
        } else {
            // Different candidate, start over
            self.candidate_intent = predicted_intent;
            self.candidate_count = 1;
        }
        self.current_intent
    }
}

struct ActionState {
    hold_counts: Vec<usize>,
    active: Vec<bool>,
}

impl ActionState {
    fn new(num_keys: usize) -> Self {
        Self {
            hold_counts: vec![0; num_keys],
            active: vec![false; num_keys],
        }
    }

    fn update(
        &mut self,
        logits: &[f32],
        threshold: f32,
        debounce: usize,
        max_keys: usize,
    ) -> ActionDelta {
        let probs: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, v)| (i, 1.0 / (1.0 + (-v).exp())))
            .collect();

        let mut desired = vec![false; logits.len()];
        for (idx, prob) in &probs {
            desired[*idx] = *prob > threshold;
        }

        if max_keys > 0 {
            let active_count = desired.iter().filter(|v| **v).count();
            if active_count > max_keys {
                let mut active_probs: Vec<(usize, f32)> = probs
                    .iter()
                    .filter(|(idx, _)| desired[*idx])
                    .cloned()
                    .collect();
                active_probs
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                desired.fill(false);
                for (idx, _prob) in active_probs.iter().take(max_keys) {
                    desired[*idx] = true;
                }
            }
        }

        let mut pressed = Vec::new();
        let mut released = Vec::new();

        for idx in 0..desired.len() {
            let is_active = desired[idx];
            let prev_active = self.active[idx];
            if is_active == prev_active {
                self.hold_counts[idx] = 0;
                continue;
            }
            self.hold_counts[idx] += 1;
            if self.hold_counts[idx] >= debounce {
                self.active[idx] = is_active;
                if is_active {
                    pressed.push(idx);
                } else {
                    released.push(idx);
                }
                self.hold_counts[idx] = 0;
            }
        }

        ActionDelta {
            pressed,
            released,
            active: self.active.clone(),
        }
    }
}

struct ActionDelta {
    pressed: Vec<usize>,
    released: Vec<usize>,
    #[allow(dead_code)]
    active: Vec<bool>,
}

fn load_manifest_path(model: &str, override_path: &Option<String>) -> String {
    if let Some(path) = override_path {
        return path.to_string();
    }
    let base = model.trim_end_matches(".onnx");
    format!("{base}_manifest.json")
}

fn capture_live_frame(height: usize, width: usize, window: &Option<String>) -> Result<Vec<f32>> {
    let mut options = ScreenshotOptions::default();
    if let Some(title) = window {
        if let Some(win) = find_window(title)? {
            options.crop_region = Some(win.region);
        } else {
            return Err(anyhow!("window not found: {}", title));
        }
    }
    let rgba = capture_screenshot_image(options)?;
    let img = DynamicImage::ImageRgba8(rgba);
    let resized = image::imageops::resize(&img, width as u32, height as u32, FilterType::Triangle);
    let mut out = Vec::with_capacity(height * width * 3);
    for pixel in resized.pixels() {
        out.push(pixel[0] as f32 / 255.0);
        out.push(pixel[1] as f32 / 255.0);
        out.push(pixel[2] as f32 / 255.0);
    }
    Ok(out)
}

fn build_pixels(height: usize, width: usize, frames: &VecDeque<Vec<f32>>) -> Array4<f32> {
    let mut pixels = Array4::<f32>::zeros((1, 9, height, width));
    for (stack_idx, frame_data) in frames.iter().enumerate() {
        for y in 0..height {
            for x in 0..width {
                let base = (y * width + x) * 3;
                for c in 0..3 {
                    pixels[[0, stack_idx * 3 + c, y, x]] = frame_data[base + c];
                }
            }
        }
    }
    pixels
}

fn map_key_name(name: &str) -> Option<Key> {
    let trimmed = name.trim();
    let key_name = trimmed
        .strip_prefix("KEY_")
        .unwrap_or(trimmed)
        .to_lowercase();
    let mapped = match key_name.as_str() {
        "leftshift" => "lshift",
        "rightshift" => "rshift",
        "leftctrl" => "lctrl",
        "rightctrl" => "rctrl",
        "leftalt" => "lalt",
        "rightalt" => "ralt",
        "esc" => "esc",
        "enter" => "enter",
        "pageup" => "pageup",
        "pagedown" => "pagedown",
        "kp0" => return Some(Key::KEY_KP0),
        "kp1" => return Some(Key::KEY_KP1),
        "kp2" => return Some(Key::KEY_KP2),
        "kp3" => return Some(Key::KEY_KP3),
        "kp4" => return Some(Key::KEY_KP4),
        "kp5" => return Some(Key::KEY_KP5),
        "kp6" => return Some(Key::KEY_KP6),
        "kp7" => return Some(Key::KEY_KP7),
        "kp8" => return Some(Key::KEY_KP8),
        "kp9" => return Some(Key::KEY_KP9),
        "kpdot" => return Some(Key::KEY_KPDOT),
        "kpplus" => return Some(Key::KEY_KPPLUS),
        "kpminus" => return Some(Key::KEY_KPMINUS),
        "kpslash" => return Some(Key::KEY_KPSLASH),
        "kpasterisk" => return Some(Key::KEY_KPASTERISK),
        other => other,
    };
    inputctl::parse_key_name(mapped).ok()
}

fn live_loop(
    cli: &Cli,
    engine: &mut InferenceEngine,
    window: &Option<String>,
    fps: f32,
) -> Result<()> {
    // Initial intent from CLI or default to first
    let initial_intent_idx = cli
        .goal_intent
        .as_ref()
        .and_then(|name| engine.intent_index(name))
        .unwrap_or(0);

    let mut goal = engine.one_hot_goal(initial_intent_idx);
    let mut intent_state = IntentState::new(initial_intent_idx, cli.intent_stability);
    let mut buffer: VecDeque<Vec<f32>> = VecDeque::with_capacity(3);
    let mut action_state = ActionState::new(engine.manifest.outputs.keys_logits.keys.len());
    let mut input = InputCtl::new()?;

    let frame_interval = if fps > 0.0 {
        Duration::from_secs_f32(1.0 / fps)
    } else {
        Duration::from_secs_f32(0.0)
    };

    if cli.verbose {
        println!(
            "Starting live loop: auto_intent={}, initial_intent={:?}",
            cli.auto_intent,
            engine.intent_name(initial_intent_idx)
        );
    }

    loop {
        let start = Instant::now();
        let frame = capture_live_frame(cli.height, cli.width, window)?;
        if buffer.len() == 3 {
            buffer.pop_front();
        }
        buffer.push_back(frame);
        while buffer.len() < 3 {
            buffer.push_front(vec![0.0; cli.height * cli.width * 3]);
        }

        let pixels = build_pixels(cli.height, cli.width, &buffer);
        let (keys_logits, intent_logits, value) = engine.run(&pixels, &goal)?;

        // Auto-intent: use predicted intent as goal for next frame
        if cli.auto_intent {
            let predicted_intent_idx = intent_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let stable_intent = intent_state.update(predicted_intent_idx);
            goal = engine.one_hot_goal(stable_intent);

            if cli.verbose {
                println!(
                    "value={:.3} pred_intent={:?} stable_intent={:?}",
                    value,
                    engine.intent_name(predicted_intent_idx),
                    engine.intent_name(stable_intent)
                );
            }
        } else if cli.verbose {
            println!("value={:.3}", value);
        }

        let delta = action_state.update(&keys_logits, cli.threshold, cli.debounce, cli.max_keys);

        for idx in delta.released {
            if let Some(name) = engine.manifest.outputs.keys_logits.keys.get(idx) {
                if let Some(key) = map_key_name(name) {
                    let _ = input.key_up(key);
                }
            }
        }
        for idx in delta.pressed {
            if let Some(name) = engine.manifest.outputs.keys_logits.keys.get(idx) {
                if let Some(key) = map_key_name(name) {
                    let _ = input.key_down(key);
                }
            }
        }

        let elapsed = start.elapsed();
        if frame_interval > elapsed {
            thread::sleep(frame_interval - elapsed);
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let manifest_path = load_manifest_path(&cli.model, &cli.manifest);
    let mut engine = InferenceEngine::load(&cli.model, &manifest_path)?;

    match &cli.command {
        CommandMode::Live { window, fps } => live_loop(&cli, &mut engine, window, *fps),
    }
}
