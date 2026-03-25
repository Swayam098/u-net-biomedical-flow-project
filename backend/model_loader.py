import torch
from backend.inference import UNet
import os

def load_unet_model(model_path):
    device = torch.device("cpu")
    model = UNet()

    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print("✅ Trained model loaded")
        except Exception as e:
            print("⚠️ Model file invalid, using untrained model")
            print(e)
    else:
        print("⚠️ Model file not found, using untrained model")

    model.eval()
    return model
