from abc import ABC, abstractmethod


class DetectBase(ABC):

    @staticmethod
    @abstractmethod
    def forward_handle_input(color_frame, depth_frame):
        pass

    @staticmethod
    @abstractmethod
    def gen_model():
        pass

    @staticmethod
    @abstractmethod
    def backward_handle_output(output, color_img, depth_img, input):
        pass

    @staticmethod
    @abstractmethod
    def detect(model, color_img, deep_data3, conf):
        pass

    @staticmethod
    @abstractmethod
    def delete_model(model):
        pass
