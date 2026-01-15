import torch
import torch.nn as nn
import numpy as np


class MyModel(nn.Module):
    """
    CNN model used on Jetson-style nodes (CIFAR10 default).
    The network is designed to be stable with BatchNorm and progressive pooling.
    The classifier input size is computed dynamically to support different input sizes.
    """

    def __init__(self, input_channels=3, input_size=32):
        super().__init__()

        # -----------------------------------------------------------------
        # Input configuration
        # -----------------------------------------------------------------
        # input_channels: 3 for CIFAR10 (RGB), 1 for MNIST (grayscale).
        # input_size:     32 for CIFAR10, 28 for MNIST (or resized inputs).
        self.input_channels = input_channels

        # -----------------------------------------------------------------
        # Feature extractor: 3 Conv blocks + BatchNorm + ReLU + MaxPool
        # -----------------------------------------------------------------
        # Each pooling step halves the spatial resolution.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # -----------------------------------------------------------------
        # Dynamic flatten-size computation
        # -----------------------------------------------------------------
        # Computes the correct Linear input dimension for the chosen input_size.
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            out = self.feature_extractor(dummy)
            flattened_size = out.view(1, -1).size(1)

        # -----------------------------------------------------------------
        # Classifier head
        # -----------------------------------------------------------------
        # Flattens the extracted feature map and outputs 10 classes.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        # -----------------------------------------------------------------
        # Forward pass: feature extraction then classification
        # -----------------------------------------------------------------
        x = self.feature_extractor(x)
        return self.classifier(x)

    # ---------------------------------------------------------------------
    # Delta adaptation for heterogeneous DFL
    # ---------------------------------------------------------------------
    # convert_weights adapts incoming DELTAS to match this model's parameter
    # shapes. Missing parameters are treated as zero deltas.
    # If shapes mismatch but tensor ranks match, floating-point tensors are
    # projected into the target shape.
    def convert_weights(self, incoming_deltas_dict):
        adapted_deltas = {}
        current_state = self.state_dict()
        device = next(self.parameters()).device

        for name, current_param in current_state.items():
            # If the sender does not provide this parameter delta, treat as zero
            if name not in incoming_deltas_dict:
                adapted_deltas[name] = torch.zeros_like(current_param, device=device)
                continue

            incoming = incoming_deltas_dict[name].detach().to(device)
            target_shape = current_param.shape

            # Direct shape match: apply delta as-is
            if incoming.shape == target_shape:
                adapted_deltas[name] = incoming

            # Same number of dimensions + float tensors: project to target shape
            elif (
                len(incoming.shape) == len(target_shape)
                and incoming.dtype.is_floating_point
                and current_param.dtype.is_floating_point
            ):
                projected = self._project_tensor(incoming, target_shape)
                if projected is None:
                    adapted_deltas[name] = torch.zeros_like(current_param, device=device)
                else:
                    adapted_deltas[name] = projected.to(device)

            # Incompatible shapes or non-float parameters: ignore and use zero delta
            else:
                adapted_deltas[name] = torch.zeros_like(current_param, device=device)

        # Ensure every parameter key exists in the returned dict
        final_adapted = {}
        for name, param in current_state.items():
            final_adapted[name] = adapted_deltas.get(
                name, torch.zeros_like(param, device=device)
            ).to(device)

        return final_adapted

    # ---------------------------------------------------------------------
    # Projection routine for mismatched tensor shapes
    # ---------------------------------------------------------------------
    # If incoming tensor has enough elements, it is randomly projected into
    # target_size using a random Gaussian matrix. If it has fewer elements,
    # it is padded with zeros. Returns None if invalid values appear.
    @staticmethod
    def _project_tensor(tensor, target_shape):
        try:
            flat = tensor.view(-1)
            target_size = int(np.prod(target_shape))

            if flat.numel() >= target_size:
                proj = torch.randn(target_size, flat.numel(), device=flat.device) * (
                    1.0 / np.sqrt(target_size)
                )
                out = proj @ flat
                out = torch.nan_to_num(out, nan=0.0, posinf=1e5, neginf=-1e5)
            else:
                pad = torch.zeros(
                    target_size - flat.numel(), device=tensor.device, dtype=tensor.dtype
                )
                out = torch.cat([flat, pad])

            if torch.isnan(out).any() or torch.isinf(out).any():
                return None

            return out.view(target_shape)

        except Exception:
            return None
