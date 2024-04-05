import numpy as np
import torch, os, io, cv2, boto3
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    env = os.environ
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

        # Get S3 access key and secret access key from env
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        bucket_name = 'your_bucket_name'
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

        # Upload image to S3 bucket
        s3.upload_fileobj(buffer, bucket_name, 'result.jpg')

    # Return JSON object indicating the image was uploaded
    return {"message": "Image uploaded successfully to S3"}