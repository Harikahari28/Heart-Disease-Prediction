import torch
import torch.nn.functional as F
from .model import EchoNetClassifier
from .video_dataset import EchoVideoProcessor

DEVICE = torch.device("cpu")

def load_echo_model():
    model = EchoNetClassifier().to(DEVICE)

    checkpoint = torch.load(
        "video_model/best_model.pth",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

@torch.no_grad()
def predict_video(video_path, model):
    processor = EchoVideoProcessor()
    clip = processor.preprocess(video_path).to(DEVICE)

    logits = model(clip)                 # shape (1, 2)
    probs = F.softmax(logits, dim=1)     # softmax
    disease_prob = probs[0, 1].item()    # class 1 = HF

    return disease_prob
