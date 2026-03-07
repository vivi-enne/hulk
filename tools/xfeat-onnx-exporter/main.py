import onnxruntime
import torch
import torch.nn.functional as F


class InterpolateSparse2d(torch.nn.Module):
    def __init__(
        self, mode: str = "bicubic", align_corners: bool = False
    ) -> None:
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(
        self,
        coordinates: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        # Convert dimensions to tensors to preserve dynamic axes during ONNX export
        width_tensor = torch.tensor(
            width, dtype=coordinates.dtype, device=coordinates.device
        )
        height_tensor = torch.tensor(
            height, dtype=coordinates.dtype, device=coordinates.device
        )

        x_coordinates = coordinates[..., 0] / (width_tensor - 1.0)
        y_coordinates = coordinates[..., 1] / (height_tensor - 1.0)

        normalized_coordinates = torch.stack(
            [x_coordinates, y_coordinates], dim=-1
        )
        return 2.0 * normalized_coordinates - 1.0

    def forward(
        self,
        feature_tensor: torch.Tensor,
        positions: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        grid = (
            self.normgrid(positions, height, width)
            .unsqueeze(-2)
            .to(feature_tensor.dtype)
        )
        sampled = F.grid_sample(
            feature_tensor, grid, mode=self.mode, align_corners=False
        )
        return sampled.permute(0, 2, 3, 1).squeeze(-2)


class XFeatOnnx(torch.nn.Module):
    def __init__(self, top_k: int = 512, threshold: float = 0.1) -> None:
        super().__init__()
        self.top_k = top_k
        self.threshold = threshold
        self.model = torch.hub.load(
            "verlab/accelerated_features",
            "XFeat",
            pretrained=True,
            trust_repo="check",
        )

    def non_maximum_suppression(
        self,
        keypoint_heatmap: torch.Tensor,
    ) -> torch.Tensor:
        kernel_size = 5
        padding = kernel_size // 2

        batch_size = keypoint_heatmap.size(0)
        height = keypoint_heatmap.size(2)
        width = keypoint_heatmap.size(3)

        local_maxima = F.max_pool2d(
            keypoint_heatmap, kernel_size=kernel_size, stride=1, padding=padding
        )

        is_peak = (keypoint_heatmap == local_maxima) & (
            keypoint_heatmap > self.threshold
        )
        suppressed_heatmap = keypoint_heatmap * is_peak

        suppressed_heatmap_flat = suppressed_heatmap.view(batch_size, -1)

        _, top_indices = torch.topk(suppressed_heatmap_flat, self.top_k, dim=1)

        # Convert width to a tensor to ensure ONNX mathematical operations do not attempt to cast SymInts to pure Python integers
        width_tensor = torch.tensor(
            width, dtype=top_indices.dtype, device=top_indices.device
        )

        y_coordinates = torch.div(
            top_indices, width_tensor, rounding_mode="floor"
        )
        x_coordinates = torch.remainder(top_indices, width_tensor)

        positions = torch.stack([x_coordinates, y_coordinates], dim=-1)

        return positions

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height = image.size(2)
        width = image.size(3)

        feature_map, keypoint_logits, heatmap_base = self.model.net(image)
        feature_map = F.normalize(feature_map, dim=1)
        keypoint_heatmap = self.model.get_kpts_heatmap(keypoint_logits)

        keypoints = self.non_maximum_suppression(keypoint_heatmap)

        nearest = InterpolateSparse2d("nearest")
        bilinear = InterpolateSparse2d("bilinear")

        scores = (
            nearest(keypoint_heatmap, keypoints, height, width)
            * bilinear(heatmap_base, keypoints, height, width)
        ).squeeze(-1)

        is_zero_keypoint = torch.all(keypoints == 0, dim=-1)
        scores = torch.where(
            is_zero_keypoint,
            torch.tensor(-1.0, device=scores.device, dtype=scores.dtype),
            scores,
        )

        sorted_indices = torch.argsort(-scores)

        keypoints_x = torch.gather(keypoints[..., 0], 1, sorted_indices)
        keypoints_y = torch.gather(keypoints[..., 1], 1, sorted_indices)
        keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)

        scores = torch.gather(scores, 1, sorted_indices)

        descriptors = bilinear(feature_map, keypoints, height, width)
        descriptors = F.normalize(descriptors, dim=-1)

        return keypoints, scores, descriptors


@torch.inference_mode()
def main() -> None:
    model = XFeatOnnx(top_k=256, threshold=0.2)
    image = torch.randn(2, 1, 480, 640)

    torch.onnx.export(
        model,
        (image,),
        "xfeat.onnx",
        input_names=["input"],
        output_names=["keypoints", "scores", "descriptors"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "keypoints": {0: "batch_size"},
            "scores": {0: "batch_size"},
            "descriptors": {0: "batch_size"},
        },
        do_constant_folding=True,
        opset_version=22,
    )

    session = onnxruntime.InferenceSession("xfeat.onnx")
    out = session.run(
        ["keypoints", "scores", "descriptors"],
        {
            "input": image.numpy(),
        },
    )


if __name__ == "__main__":
    main()
