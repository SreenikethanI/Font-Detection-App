import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import numpy as np
import PIL.Image as Image

ALNUM = "QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm0123456789"

transform_image_no_affine = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True), # Converts [0, 255] to [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,)), # Converts [0.0, 1.0] to [-1.0, 1.0]
])

def load_fontnames(path: str) -> list[str]:
    """Load the font names from the given text file. Lines starting with a `#`
    are ignored."""

    with open(path, "r") as f:
        return [ls for line in f.readlines() if not (ls := line.strip()).startswith("#") and ls]

class FontIdentificationModel(nn.Module):
    def __init__(self, num_fonts, largest_char_code: int, char_code_dim=64):
        super(FontIdentificationModel, self).__init__()

        # CNN https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
        # For Conv2d layers, we neither use stride>1 nor dilation, since the
        # images are already low in resolution.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.3)

        # The sequence of layers below are as described in "Sequence 1" above.
        self.layer1_conv = nn.Conv2d(1, 128, kernel_size=15, padding=7) # the 1 input stands for 1-channel image i.e. grayscale.
        self.layer1_conv_bn = nn.BatchNorm2d(128)
        self.layer2_conv = nn.Conv2d(128, 64, kernel_size=15, padding=7)
        self.layer2_conv_bn = nn.BatchNorm2d(64)
        self.layer3_conv = nn.Conv2d(64, 32, kernel_size=7, padding=3)
        self.layer3_conv_bn = nn.BatchNorm2d(32)
        self.layer4_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer4_conv_bn = nn.BatchNorm2d(32)

        # The sequence of (one) layer below is as described in "Sequence 2" above.
        self.char_code_embedding = nn.Embedding(largest_char_code+10, char_code_dim)

        # The above two sequences are concatenated (in `forward()`) and then fed
        # to regular NN layers.
        self.layer5_fc = nn.Linear(32 * 8 * 8 + char_code_dim, 256)
        self.layer6_fc = nn.Linear(256, num_fonts)

    def forward(self, image, char_code):
        # Image feature extraction
        x = self.dropout(self.layer1_conv_bn(self.pool(F.relu(self.layer1_conv(image)))))
        x = self.dropout(self.layer2_conv_bn(self.pool(F.relu(self.layer2_conv(x)))))
        x = self.dropout(self.layer3_conv_bn(self.pool(F.relu(self.layer3_conv(x)))))
        x = self.layer4_conv_bn(F.relu(self.layer4_conv(x)))
        x = torch.flatten(x, 1)
        # x = x.view(x.size(0), -1)  # Flatten in-place(?)

        # Character code embedding
        char_features = self.char_code_embedding(char_code)

        # Concatenate Sequence 1 and Sequence 2 as described above.
        combined = torch.cat((x, char_features), dim=1)

        # Fully connected layers
        x = F.relu(self.layer5_fc(combined))
        x = self.layer6_fc(x)
        # x = self.layer6_softmax(x)
        return x

class SimbleModel:
    def __init__(self, model_path: str, font_names_path: str):
        font_names = load_fontnames(font_names_path)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_reverse_mapping = sorted(font_names)
        self.model = FontIdentificationModel(len(font_names), ord(max(ALNUM)), char_code_dim=128)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def predict(self, img: Image.Image, char_code: int) -> list[tuple[str, float]]:
        img_transformed = transform_image_no_affine(torch.tensor(np.array(img))).unsqueeze(dim=0)
        with torch.no_grad():
            outputs = nn.Softmax(dim=0)(self.model(
                img_transformed.to(self.device),
                torch.tensor([char_code]).to(self.device),
            ).squeeze(dim=0))

        return sorted([
            (self.label_reverse_mapping[i], score*100)
            for i, score in enumerate(outputs.tolist())
        ], key=lambda x: (-x[1], x[0]))
