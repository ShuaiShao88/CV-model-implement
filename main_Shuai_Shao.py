from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from io import BytesIO
from torchvision import transforms
from torchvision.models.resnet import ResNet18_Weights

# https://pytorch.org/hub/pytorch_vision_resnet/
model = torch.hub.load('pytorch/vision:v0.10.0',
                       'resnet18',
                       weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()  #turn off dropout and batch normalization

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

###################
## API ENDPOINTS ##
###################
app = FastAPI()


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # await: file uploads take time, so server can handle other requests while waiting for upload to finish
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    # Convert image to the format expected by the model
    preprocess = transforms.Compose([
        transforms.Resize(256),  #resize shorter side to 256
        transforms.CenterCrop(224),  #center crop of 224x224
        transforms.ToTensor(),  # Changes format from (H,W,C) to (C,H,W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = preprocess(image)  # applies all transforms
    image_tensor = image_tensor.unsqueeze(
        0)  #adds a batch dimension -> from (3,224,224) to (1,3,224,224)

    with torch.no_grad():  #turn off gradients
        output = model(image_tensor)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # takes the first and only batch item [0] and applies softmax across the class dimension (column)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Top 5 categories per image (5 highest probabilities)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Create a dictionary to store the results
    results = {}
    for i in range(top5_prob.size(0)):
        category = categories[
            top5_catid[i]]  # Get the category name using the catid index
        probability = top5_prob[i].item(
        )  # Get the probability as a Python float
        results[category] = probability

    return JSONResponse(content={"detections": results})
