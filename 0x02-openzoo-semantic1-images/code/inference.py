import numpy as np
import os, io, cv2, base64
from io import BytesIO
import openvino as ov

core = ov.Core()

def resize(mask, height, width):
    # Resize the array to match the height of the target shape (1280, 720)
    resized_array = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_array

def preprocess(frame, H, W):
    """
    Preprocess the frame for openvino model.
    """
    image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(image_bgr, (W, H))
    # Reshape to the network input shape.
    input_image = np.expand_dims(
        resized_image.transpose(2, 0, 1), 0
    )  
    return input_image

def postprocess(frame, model_result, height, width, alpha = 0.3):
    """
    Postprocess the frame for visualization.
    """
    # Constants
    colormap = np.array([[0, 0, 0], [119, 255, 51], [120, 183, 53], [86, 210, 245]])
    # Divide the mask into four equal parts along the first axis
    parts = np.split(model_result, 4, axis=1)
    # Select the road mask and curbs parts
    bg = parts[0].reshape((512,896,1))
    road = parts[1].reshape((512,896,1))
    marks = parts[2].reshape((512,896,1))
    curbs = parts[3].reshape((512,896,1))
    # Rescale the values to the range [0, 255]
    bg_white_mask = bg * [255, 255, 255]
    road_white_mask = road * [255, 255, 255]
    curbs_white_mask = curbs * [255, 255, 255]
    marks_white_mask = marks * [255, 255, 255]
    road_colored_mask = road * colormap[1]
    marks_colored_mask = marks * colormap[2]
    curbs_colored_mask = curbs * colormap[3]
    # Change type of np array to avoid errors later
    bg_white_mask = bg_white_mask.astype('uint8')
    road_colored_mask = road_colored_mask.astype('uint8')
    marks_colored_mask = marks_colored_mask.astype('uint8')
    curbs_colored_mask = curbs_colored_mask.astype('uint8')
    road_white_mask = road_white_mask.astype('uint8')
    marks_white_mask = marks_white_mask.astype('uint8')
    curbs_white_mask= curbs_white_mask.astype('uint8')
    # Resize to original image width and height
    bg_white_mask = resize(bg_white_mask, height, width)
    road_white_mask = resize(road_white_mask, height, width)
    marks_white_mask = resize(marks_white_mask, height, width)
    curbs_white_mask= resize(curbs_white_mask, height, width)
    road_colored_mask = resize(road_colored_mask, height, width)
    marks_colored_mask = resize(road_colored_mask, height, width)
    curbs_colored_mask = resize(curbs_colored_mask, height, width)
    # Image Arthimatic Operations
    colored_mask = road_colored_mask + curbs_colored_mask
    white_mask = road_white_mask + curbs_white_mask + marks_white_mask
    # Bitwise Operations
    subtracted_road=cv2.bitwise_and(frame, bg_white_mask,mask=None)
    segmented_road=cv2.bitwise_and(frame, white_mask,mask=None)
    # Overlay
    alpha = .7
    overlay = segmented_road.copy()
    segmented_image = cv2.addWeighted(colored_mask, alpha, overlay, 1-alpha, 0, overlay)
    final_result=cv2.bitwise_or(subtracted_road,segmented_image,mask=None)
    
    return final_result

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    model_path = os.path.join(model_dir, "code", "road-segmentation-adas-0001.xml")
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model=model, device_name='CPU')
    input_layer_ir = compiled_model.input(0)
    output_layer_ir = compiled_model.output(0)
    N, C, H, W = input_layer_ir.shape
    # return all the previous variables as a dict called model
    result = {
        "model": compiled_model,
        "height": H,
        "width": W,
        "output_layer_ir": output_layer_ir
    }
    return result


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
    # Extract the model variables from the model dictionary
    compiled_model = model["model"]
    H = model["height"]
    W = model["width"]
    output_layer_ir = model["output_layer_ir"]
    # Preprocess the input image
    height, width, _ = input_data.shape
    preprocessed_image = preprocess(input_data, H, W)
    # Perform inference
    output_image = compiled_model([preprocessed_image])[output_layer_ir]
    # Post-processing steps here...
    postprocessed_image = postprocess(input_data, output_image, height, width)
    return postprocessed_image
       
def output_fn(prediction_output, content_type):
    print("Executing output_fn from inference.py ...")
    img = prediction_output
    # Convert the image to a BytesIO buffer
    _, buffer = cv2.imencode('.jpg', img)
    # Create a BytesIO object from the buffer
    buffer = BytesIO(buffer)
    # Convert BytesIO to base64-encoded string
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Return the image data as a JSON-serializable response
    return {'image': image_data}