from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, kw_only=True)
class ExtractedFeatures:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray

    def __getitem__(
        self, index: slice | np.ndarray | int
    ) -> "ExtractedFeatures":
        return ExtractedFeatures(
            keypoints=self.keypoints[index],
            descriptors=self.descriptors[index],
            scores=self.scores[index],
        )


class FeatureDetectMatcher:
    def __init__(self) -> None:
        self.model = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=128
        )

    def extract(self, image: np.ndarray, topk: int = 128) -> ExtractedFeatures:
        tensor = torch.from_numpy(image).to(torch.float32) / 255.0
        input_image = tensor[None, None, ..., 0]
        outputs = self.model.detectAndCompute(input_image, top_k=topk)[0]
        return ExtractedFeatures(
            keypoints=outputs["keypoints"].numpy(),
            scores=outputs["scores"].numpy(),
            descriptors=outputs["descriptors"].numpy(),
        )

    def match(
        self,
        features_source: ExtractedFeatures,
        features_target: ExtractedFeatures,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.model.match(
            torch.from_numpy(features_source.descriptors),
            torch.from_numpy(features_target.descriptors),
        )
