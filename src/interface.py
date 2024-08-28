import torch
import base
from model import network
import inference_core
import interactive_utils


class TrackRmx(base.DetectBase):

    @staticmethod
    def gen_model(weight, device="cuda"):
        device = torch.device(device)
        auto_collection_config = {
            "workspace": None,
            "buffer_size": 10,
            "max_mid_term_frames": 10,
            "min_mid_term_frames": 5,
            "max_long_term_elements": 10000,
            "num_prototypes": 128,
            "top_k": 30,
            "mem_every": 10,
            "deep_update_every": -1,
            "no_amp": False,
            "size": 480,
            "enable_long_term": True,
            "enable_long_term_count_usage": True,
            "key_dim": 64,
            "value_dim": 512,
            "hidden_dim": 64,
        }
        net = (
            network.XMem(auto_collection_config, weight, map_location=device)
            .to(device)
            .eval()
        )
        processor = inference_core.InferenceCore(net, auto_collection_config)
        return processor

    @staticmethod
    def forward_handle_input():
        pass

    @staticmethod
    def backward_handle_output():
        pass

    @staticmethod
    def detect(processor, mark_image, mark_mask, predict_image, num_objects, device='cuda'):
        processor.set_all_labels(list(range(1, num_objects + 1)))
        device = torch.device(device)

        # 加载归一化张量图像与未归一化张量图像
        (
            mark_image_torch,
            mark_image_torch_no_norm,
        ) = interactive_utils.image_to_torch(mark_image, device)

        mark_prob = interactive_utils.index_numpy_to_one_hot_torch(
            mark_mask, num_objects + 1
        ).to(device)

        mark_prob = processor.step(mark_image_torch, mark_prob[1:])
        mark_mask = interactive_utils.torch_prob_to_numpy_mask(mark_prob)

        predict_image_torch = None

        (
            predict_image_torch,
            predict_image_torch_no_norm,
        ) = interactive_utils.image_to_torch(predict_image, device)

        predict_prob = processor.step(predict_image_torch)
        predict_mask = interactive_utils.torch_prob_to_numpy_mask(
            predict_prob
        )

        try:
            max_work_elements = processor.memory.max_work_elements
            max_long_elements = processor.memory.max_long_elements

            curr_work_elements = processor.memory.work_mem.size
            curr_long_elements = processor.memory.long_mem.size

        except AttributeError:
            pass

        return predict_mask

    @staticmethod
    def delete_model(
            processor,
            device="cuda",
    ):
        if not processor is None:
            processor.clear_memory()
        if device == "cuda":
            torch.cuda.empty_cache()
