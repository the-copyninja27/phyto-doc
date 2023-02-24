import logging
import azure.functions as func
from PIL import Image
import onnxruntime
import torchvision
from .model_attributes import train_classes
import numpy as np

def main(req: func.HttpRequest) -> func.HttpResponse:
    ort_session = onnxruntime.InferenceSession("models/plant-disease.onnx")
    transformer = torchvision.transforms.ToTensor()
    crop_image = req.files.get('crop_image')
    logging.info(crop_image)
    if not crop_image:
        return func.HttpResponse("Check image and reupload", status_code=400)
    pillow_obj = Image.open(crop_image)
    resized_image = pillow_obj.resize((256, 256), Image.LANCZOS)
    image_tensor = transformer(resized_image)
    image_np = image_tensor.unsqueeze(0).cpu().detach().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: image_np}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_result = train_classes[int(np.argmax(ort_outs[0], axis=1))]
    disease = onnx_result
    if disease:
        return func.HttpResponse(disease)
    return func.HttpResponse("Intenal Server Error", status_code=500)
