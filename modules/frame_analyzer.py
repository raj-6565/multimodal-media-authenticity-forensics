import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import cv2

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def analyze_frame(frame_path):
    """
    Analyze frame and return anomaly score.
    """

    try:
        frame = cv2.imread(frame_path)

        if frame is None:
            return 0.0

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_tensor = transform(frame_rgb).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        confidence = float(torch.max(probabilities))

        anomaly_score = 1 - confidence

        return anomaly_score

    except Exception:
        return 0.0
