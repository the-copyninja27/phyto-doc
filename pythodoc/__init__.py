import logging
import azure.functions as func
from PIL import Image
import onnxruntime
import torchvision
from .model_attributes import train_classes


def main(req: func.HttpRequest) -> func.HttpResponse:
    ort_session = onnxruntime.InferenceSession("models/plant-disease.onnx")
    logging.info('Python HTTP trigger function processed a request.')
    name = req.params.get('name')
    return func.HttpResponse("onnx runtime and torch vision good to go")
