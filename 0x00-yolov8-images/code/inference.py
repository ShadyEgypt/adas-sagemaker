import numpy as np
import torch, os, io, cv2, base64
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    model_path = os.path.join(model_dir, "code", "best.pt")
    model = YOLO(model_path)
    return model


def input_fn(request_body, request_content_type):
    print("Executing input_fn from inference.py ...")
    if request_content_type:
        jpg_original = np.load(io.BytesIO(request_body), allow_pickle=True)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=-1)
    else:
        raise Exception("Unsupported content type: " + request_content_type)
    return img
    
def predict_fn(input_data, model):
    print("Executing predict_fn from inference.py ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        result = model(input_data)
    return result
       
def output_fn(prediction_output, content_type):
    print("Executing output_fn from inference.py ...")
    for r in prediction_output:
        im_array = r.plot(probs=False,conf=False, boxes=False) 
        im = Image.fromarray(im_array[..., ::-1])

        # Upload image to S3 bucket
        buffer = BytesIO()
        im.save(buffer, format='JPEG')
        buffer.seek(0)

        # Convert BytesIO to base64-encoded string
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Return the image data as a JSON-serializable response
        return {'image': image_data}