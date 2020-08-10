from .base_model import BaseModel
from .ganimation import GANimationModel


def create_model(opt):
    instance = GANimationModel()
    instance.initialize(opt)
    instance.setup()
    return instance

