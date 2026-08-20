from abc import ABC, abstractmethod

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config

from detectron2.engine import DefaultPredictor


class Processer(ABC):
    @abstractmethod
    def load_model(self):
        pass
    @abstractmethod
    def preprocess(self):
        pass
    @abstractmethod
    def infer(self):
        pass
    @abstractmethod
    def postprocess(self):
        pass
    @abstractmethod
    def save(self):
        pass

    def run(self):
        self.load_model()
        self.preprocess()
        self.infer()
        self.postprocess()
        self.save()

cfg_dict = dict(
    maskdino_swinl_2048_4f=dict(
        config_path='output/maskdino_swinl-time-10-12-10-30/config.yaml',
        checkpoint_path='output/maskdino_swinl-time-10-12-10-30/model_final.pth'
    ),
    maskdino_dinov3_4096_4f=dict(
        config_path='output/maskdino_dinov3_7b16-time-10-11-15-30/config.yaml',
        checkpoint_path='output/maskdino_dinov3_7b16-time-10-11-15-30/model_final.pth'
    )
)

class Model(ABC):
    def __init__(self):
        pass

class MaskDinoModel(Model):
    def __init__(self, model_name='maskdino_swinl_2048_4f'):
        self.model_name = model_name
        self.cfg = self.get_model_config(self.model_name)
        self.model = DefaultPredictor(self.cfg)

    def get_model_config(self, model_name):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        maskdino_config = cfg_dict[model_name]
        config_path = maskdino_config['config_path']
        checkpoint_path = maskdino_config['checkpoint_path']
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = checkpoint_path
        return cfg


class ModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == 'maskdino_swinl_2048_4f':
            return MaskDinoModel(model_name='maskdino_swinl_2048_4f')
        if model_name == 'maskdino_dinov3_4096_4f':
            return MaskDinoModel(model_name='maskdino_dinov3_4096_4f')
        raise ValueError(f"Model {model_name} not recognized.")


class MaskDinoProcesser(Processer):
    def load_model(self):
        self.model_name = 'maskdino_swinl_2048_4f'
        self.model = ModelFactory.get_model(self.model_name).model
    def preprocess(self):
        pass
    def infer(self):
        pass
    def postprocess(self):
        pass
    def save(self):
        pass