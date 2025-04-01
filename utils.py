import torch
from torchvision import transforms
from model import load_model, class_names
from disease_solutions import disease_solutions  # Import from the new module


model = load_model(model_path="plantDisease/Updated Plant/model/model.pth")

# Define preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image):
    img_tensor = preprocess(image).unsqueeze(0).to('cpu')
    with torch.no_grad():
        predictions = model(img_tensor)
    predicted_class_idx = torch.argmax(predictions, dim=1).item()
    predicted_label = class_names[predicted_class_idx]
    probs = torch.nn.functional.softmax(predictions, dim=1)
    predicted_prob = probs[0][predicted_class_idx].item()

    if 'healthy' in predicted_label:
        predicted_label = "Leaf is Healthy"
        predicted_prob = 1.0

    # Get treatment solution and link
    disease_info = disease_solutions.get(predicted_label, {"solution": "No treatment available.", "link": ""})
    disease_solution = disease_info["solution"]
    disease_link = disease_info["link"]
    
    return predicted_label, predicted_prob, disease_solution, disease_link
