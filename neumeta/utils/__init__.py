
from omegaconf import OmegaConf
from .get_dataloader import *
from .other_utils import *
from .hypernet_utils import *

SLIM_FLAGS = OmegaConf.create({
    'width_mult_list_train': [0.5, 0.75, 1.0],
    'width_mult_list': [0.25, 0.5, 0.75, 1.0]
})