import onnx
import torch
import torch.nn.functional as F


class InterpolateSparse2d(torch.nn.Module):
    """Efficiently interpolate tensor at given sparse 2D positions."""

    def __init__(
        self, mode: str = "bicubic", align_corners: bool = False
    ) -> None:
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Normalize coords to [-1,1]."""
        return (
            2.0
            * (
                x
                / (torch.tensor([W - 1, H - 1], device=x.device, dtype=x.dtype))
            )
            - 1.0
        )

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode=self.mode, align_corners=False)
        return x.permute(0, 2, 3, 1).squeeze(-2)


class XFeatOnnx(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.hub.load(
            "verlab/accelerated_features",
            "XFeat",
            pretrained=True,
            trust_repo="check",
        )

    def non_maximum_suppresion(
        self,
        keypoint_heatmap: torch.Tensor,
        topk: torch.Tensor,
        threshold: torch.Tensor,
    ) -> torch.Tensor:
        kernel_size = 5
        pad = kernel_size // 2
        batch_size, _, height, width = keypoint_heatmap.shape

        local_maxima = F.max_pool2d(
            keypoint_heatmap, kernel_size=kernel_size, stride=1, padding=pad
        )

        is_peak = (keypoint_heatmap == local_maxima) & (
            keypoint_heatmap > threshold
        )
        suppressed_heatmap = keypoint_heatmap * is_peak

        # Flatten the spatial dimensions to allow vectorized sorting
        suppressed_heatmap_flat = suppressed_heatmap.view(batch_size, -1)

        limit = torch.min(
            topk,
            torch.tensor(height * width, dtype=topk.dtype, device=topk.device),
        )

        # Extract indices of the highest scoring features without Python loops
        _, top_indices = torch.topk(
            suppressed_heatmap_flat, int(limit.item()), dim=1
        )

        y_coordinates, x_coordinates = torch.unravel_index(
            top_indices, (height, width)
        )

        positions = torch.stack([x_coordinates, y_coordinates], dim=-1)

        return positions

    def forward(
        self, image: torch.Tensor, topk: torch.Tensor, threshold: torch.Tensor
    ):
        _, _, height, width = image.shape
        feature_map, keypoint_logits, heatmap_base = self.model.net(image)
        feature_map = F.normalize(feature_map, dim=1)
        keypoint_heatmap = self.model.get_kpts_heatmap(keypoint_logits)
        keypoints = self.non_maximum_suppresion(
            keypoint_heatmap, topk, threshold
        )

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

        topk_int = int(topk.item()) if topk.dim() > 0 else int(topk)
        keypoints_x = torch.gather(keypoints[..., 0], -1, sorted_indices)[
            :, :topk_int
        ]
        keypoints_y = torch.gather(keypoints[..., 1], -1, sorted_indices)[
            :, :topk_int
        ]
        keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)

        scores = torch.gather(scores, -1, sorted_indices)[:, :topk_int]

        descriptors = self.model.interpolator(
            feature_map, keypoints, H=height, W=width
        )
        descriptors = F.normalize(descriptors, dim=-1)

        return keypoints, scores, descriptors


@torch.inference_mode()
def main():
    model = XFeatOnnx()
    image = torch.randn(16, 1, 640, 480)
    topk = torch.tensor(128, dtype=torch.int64)
    threshold = torch.tensor(0.1, dtype=torch.float32)

    torch.onnx.export(
        model,
        (image, topk, threshold),
        "xfeat.onnx",
        input_names=["input", "topk", "threshold"],
        output_names=["keypoints", "scores", "descriptors"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "keypoints": {0: "batch_size", 1: "num_keypoints"},
            "scores": {0: "batch_size", 1: "num_keypoints"},
            "descriptors": {0: "batch_size", 1: "num_keypoints"},
        },
        do_constant_folding=True,
        opset_version=14,
        export_params=True,
    )


if __name__ == "__main__":
    main()
