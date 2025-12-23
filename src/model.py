import torch
import torch.nn as nn


class NEU_CNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Conv layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*14*14, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Initialize weights (The "Kaiming" logic)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            # Apply Kaiming to all Convolutional layers
            if isinstance(m, nn.Conv2d):
                # 'fan_out' preserves the magnitude in the backward pass
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # BatchNorm layers start with weight 1 and bias 0
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # Linear layers use small random weights
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ROBUSTNESS CHECK: Ensure we don't pass an empty tensor
        if x.nelement() == 0:
            raise ValueError("Input tensor is empty!")

        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


# --- The "Private Playground" for testing the architecture ---
if __name__ == "__main__":
    # Create a dummy model
    model = NEU_CNN()

    # Create a dummy "image" (Batch=1, Channels=3, H=224, W=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Run a test pass
    output = model(dummy_input)

    print("--- Model Architecture Test ---")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape (logits): {output.shape}")
    print(f"Success! Model is ready for {output.size(1)} classes.")

    # Triggering the "Robustness" Error
    model = NEU_CNN()
    model.eval()  # Set to evaluation mode

    # 2. Create a "GHOST" Tensor (Empty)
    # This simulates a situation where a file was corrupted
    # and the DataLoader accidentally sent an empty array.
    ghost_tensor = torch.tensor([])

    print("--- Starting Robustness Test ---")
    print(f"Ghost Tensor Elements: {ghost_tensor.nelement()}")

    # Try to pass the empty tensor into the model
    output = model(ghost_tensor)
