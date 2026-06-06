use color_eyre::eyre::Result;
use coordinate_systems::SlamMap;
use eframe::{
    App, CreationContext, NativeOptions,
    egui::{
        CentralPanel, ColorImage, Context, TextureHandle, TextureOptions, Ui, load::SizedTexture,
    },
};
use egui_plot::{Line, Plot, PlotPoint, PlotPoints};
use linear_algebra::{IntoTransform, Pose3};
use ndarray::{ArrayView3, s};
use visual_odometry::Pipeline;

use crate::dataset::KittiOdometrySequence;

pub struct VisualOdometryGui {
    sequence: KittiOdometrySequence,
    vo: Pipeline,
    next_index: usize,
    state: Option<PreviousState>,
}

struct PreviousState {
    time: f32,
    left_image: TextureHandle,
    right_image: TextureHandle,
    poses: Vec<Pose3<SlamMap>>,
}

impl VisualOdometryGui {
    pub fn start(sequence: KittiOdometrySequence, pipeline: Pipeline) -> eframe::Result<()> {
        eframe::run_native(
            "Visual Odometry",
            NativeOptions::default(),
            Box::new(|cc| Ok(Box::new(VisualOdometryGui::new(cc, sequence, pipeline)))),
        )
    }

    fn new(_cc: &CreationContext, sequence: KittiOdometrySequence, pipeline: Pipeline) -> Self {
        Self {
            sequence,
            vo: pipeline,
            next_index: 0,
            state: None,
        }
    }

    fn make_vo_step(&mut self, ctx: &Context) -> Result<()> {
        let next = self.sequence.get(self.next_index).unwrap()?;
        self.next_index += 1;
        let left_image = load_to_image(ctx, "left_image", next.left_image.view());
        let right_image = load_to_image(ctx, "right_image", next.right_image.view());
        let update = self
            .vo
            .step(
                next.left_image.view().slice(s![.., .., 0]),
                next.right_image.view().slice(s![.., .., 0]),
            )?
            .inverse()
            .framed_transform::<SlamMap, SlamMap>();

        self.state = match self.state.take() {
            None => Some(PreviousState {
                time: next.time,
                left_image,
                right_image,
                poses: vec![Pose3::default()],
            }),
            Some(mut state) => {
                let last = state.poses.last().unwrap();
                let next = last.as_transform() * update;
                state.poses.push(next.as_pose());
                state.left_image = left_image;
                state.right_image = right_image;
                Some(state)
            }
        };
        Ok(())
    }
}

impl App for VisualOdometryGui {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| {
            // let is_pressed = true || ctx.input(|reader| reader.key_pressed(Key::Space));
            // if (is_pressed || self.state.is_none()) && self.next_index < self.sequence.len() {
            if self.next_index < self.sequence.len() {
                if let Err(error) = self.make_vo_step(ctx) {
                    for error in error.chain() {
                        ui.label(error.to_string());
                    }
                }
            }

            if let Some(state) = &self.state {
                ui.vertical(|ui| {
                    ui.label(format!("{}s", state.time));
                    ui.horizontal(|ui| {
                        ui.image(SizedTexture::from_handle(&state.left_image));
                        ui.image(SizedTexture::from_handle(&state.right_image));
                    });
                    show_poses_plot(ui, &state.poses)
                });
            }
        });
        ctx.request_repaint();
    }
}

pub fn show_poses_plot(ui: &mut Ui, poses: &[Pose3<SlamMap>]) {
    Plot::new("pose-plot").data_aspect(1.0).show(ui, |ui| {
        ui.line(Line::new(
            "poses",
            PlotPoints::Owned(
                poses
                    .iter()
                    .map(|pose| {
                        let translation = pose.position();
                        PlotPoint::new(translation.x(), translation.z())
                    })
                    .collect(),
            ),
        ));
    });
}

pub fn load_to_image(context: &Context, name: &str, image_data: ArrayView3<u8>) -> TextureHandle {
    let shape = image_data.shape();
    let height = shape[0];
    let width = shape[1];
    let channels = shape[2];

    let dimensions = [width, height];

    let contiguous_data = image_data.as_standard_layout();
    let pixel_slice = contiguous_data
        .as_slice()
        .expect("Failed to extract slice from array");

    let color_image = match channels {
        3 => ColorImage::from_rgb(dimensions, pixel_slice),
        _ => panic!("Unsupported channel count"),
    };

    context.load_texture(name, color_image, TextureOptions::default())
}
