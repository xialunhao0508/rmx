"""
这个文件定义了XMem类，它是最高级别的nn.Module接口。
在训练过程中，它被trainer.py使用。
在评估过程中，它被inference_core.py使用。

它进一步依赖于modules.py，该文件提供了子模块的详细实现。
"""


from model.aggregate import aggregate  # 导入aggregate模块
from model.modules import *  # 导入自定义的模块
from model.memory_util import *  # 导入memory_util模块


class XMem(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        """
        model_path/map_location仅在评估过程中使用
        map_location用于将保存在cuda上的模型转到cpu上
        """
        super().__init__()
        model_weights = self.init_hyperparameters(
            config, model_path, map_location
        )  # 初始化超参数，并读取模型权重

        self.single_object = config.get("single_object", False)
        print(f"Single object mode: {self.single_object}")

        self.key_encoder = KeyEncoder()  # 创建KeyEncoder实例
        self.value_encoder = ValueEncoder(
            self.value_dim, self.hidden_dim, self.single_object
        )  # 创建ValueEncoder实例

        # 从f16特征空间投影到key/value空间
        self.key_proj = KeyProjection(1024, self.key_dim)  # 创建KeyProjection实例

        self.decoder = Decoder(self.value_dim, self.hidden_dim)  # 创建Decoder实例

        if model_weights is not None:
            self.load_weights(
                model_weights, init_as_zero_if_needed=True
            )  # 加载模型权重

    def encode_key(self, frame, need_sk=True, need_ek=True):
        # 确定输入形状
        if len(frame.shape) == 5:
            # 形状为 b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # 将其展平以便将其输入到2D CNN中
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # 形状为 b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        f16, f8, f4 = self.key_encoder(frame)  # KeyEncoder处理输入帧得到f16、f8、f4特征
        key, shrinkage, selection = self.key_proj(
            f16, need_sk, need_ek
        )  # KeyProjection处理f16特征得到key、shrinkage、selection特征

        if need_reshape:
            # B*C*T*H*W
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = (
                    shrinkage.view(b, t, *shrinkage.shape[-3:])
                    .transpose(1, 2)
                    .contiguous()
                )
            if selection is not None:
                selection = (
                    selection.view(b, t, *selection.shape[-3:])
                    .transpose(1, 2)
                    .contiguous()
                )

            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])

        return key, shrinkage, selection, f16, f8, f4

    def encode_value(
            self, frame, image_feat_f16, h16, masks, is_deep_update=True
    ):
        num_objects = masks.shape[1]
        if num_objects != 1:
            others = torch.cat(
                [
                    torch.sum(
                        masks[:, [j for j in range(num_objects) if i != j]],
                        dim=1,
                        keepdim=True,
                    )
                    for i in range(num_objects)
                ],
                1,
            )
        else:
            others = torch.zeros_like(masks)

        g16, h16 = self.value_encoder(
            frame, image_feat_f16, h16, masks, others, is_deep_update
        )  # ValueEncoder处理输入帧、图像特征和状态特征得到g16、h16特征

        return g16, h16

    # 仅在训练过程中使用
    # 在测试时，这一步被MemoryManager所取代
    def read_memory(
            self,
            query_key,
            query_selection,
            memory_key,
            memory_shrinkage,
            memory_value,
    ):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2)

        affinity = get_affinity(
            memory_key, memory_shrinkage, query_key, query_selection
        )  # 计算亲和度
        memory = readout(affinity, memory_value)  # 从memory中读取信息
        memory = memory.view(
            batch_size, num_objects, self.value_dim, *memory.shape[-2:]
        )

        return memory

    def segment(
            self,
            multi_scale_features,
            memory_readout,
            hidden_state,
            selector=None,
            h_out=True,
            strip_bg=True,
    ):
        hidden_state, logits = self.decoder(
            *multi_scale_features, hidden_state, memory_readout, h_out=h_out
        )  # Decoder进行分割预测，得到隐状态和logits
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector

        logits, prob = aggregate(prob, dim=1, return_logits=True)  # 聚合结果
        if strip_bg:
            # 去除背景
            prob = prob[:, 1:]

        return hidden_state, logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == "encode_key":
            return self.encode_key(*args, **kwargs)
        elif mode == "encode_value":
            return self.encode_value(*args, **kwargs)
        elif mode == "read_memory":
            return self.read_memory(*args, **kwargs)
        elif mode == "segment":
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        """
        初始化三个超参数：key_dim，value_dim和hidden_dim
        如果提供了model_path，我们将从模型权重中加载这些参数
        然后将实际的参数更新到config中

        否则，我们从config或默认值中加载
        """
        if model_path is not None:
            # 从模型权重中加载模型和key/value/hidden维度，并通过一些技巧更新config
            model_weights = torch.load(model_path, map_location=map_location)
            self.key_dim = model_weights['key_proj.key_proj.weight'].shape[0]
            self.value_dim = model_weights['value_encoder.fuser.block2.conv2.weight'].shape[0]
            self.disable_hidden = 'decoder.hidden_update.transform.weight' not in model_weights
            if self.disable_hidden:
                self.hidden_dim = 0
            else:
                self.hidden_dim = model_weights['decoder.hidden_update.transform.weight'].shape[0] // 3
            print(
                f"从模型权重中读取超参数：C^k={self.key_dim}，C^v={self.value_dim}，C^h={self.hidden_dim}"
            )
        else:
            model_weights = None
            # 从config或默认值中加载维度
            if "key_dim" not in config:
                self.key_dim = 64
                print(f"在config中找不到key_dim。设置为默认值{self.key_dim}")
            else:
                self.key_dim = config["key_dim"]

            if "value_dim" not in config:
                self.value_dim = 512
                print(f"在config中找不到value_dim。设置为默认值{self.value_dim}")
            else:
                self.value_dim = config["value_dim"]

            if "hidden_dim" not in config:
                self.hidden_dim = 64
                print(f"在config中找不到hidden_dim。设置为默认值{self.hidden_dim}")
            else:
                self.hidden_dim = config["hidden_dim"]

            self.disable_hidden = self.hidden_dim <= 0

        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim

        return model_weights

    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    print('Converting weights from single object to multiple objects.')
                    pads = torch.zeros((64, 1, 7, 7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    else:
                        print('Zero-initialized padding.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.load_state_dict(src_dict)
