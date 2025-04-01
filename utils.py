import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the model
def load_model(model_path="Updated Plant/model/model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()  # Set to evaluation mode
    return model

# Load the model globally to avoid reloading it on every prediction
model = load_model()  # Ensure this is executed when utils.py is imported

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),
])

def predict_image(img):
    global model  # Ensure model is accessible
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        predictions = model(img_tensor)  # Get model output

    # Mock result (modify based on your actual model output processing)
    predicted_label = "Healthy"
    predicted_prob = torch.max(predictions).item()
    disease_solution = "No action needed"
    disease_link = None
    
    return predicted_label, predicted_prob, disease_solution, disease_link



    
  


