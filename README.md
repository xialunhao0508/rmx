# rmx_SDK

## **1. 项目介绍**

该模型可以根据目标物体在第一帧的掩码(mask)描述，在后续帧中跟踪该物体并生成相应的掩码(mask)
。以睿眼为例。我们再对采集的数据作自动标注的时候，只需手动标注一张图片，得到这张图片的mask，后续就可以用这个模型来实现自动追踪的功能。

- **API链接**：[API链接地址](http://192.168.0.188:8090/ai_lab_rd02/ai_sdks/xmem)

## **2. 代码结构**

```
xmem/
│
├── README.md        <- 项目的核心文档
├── requirements.txt    <- 项目的依赖列表
├── setup.py        <- 项目的安装脚本
├── .gitignore        <- 忽略文件
│
├── rmx/          <- 项目的源代码
│  ├── dataset/         <- 
│  ├── model/           <- 
│  ├── util/            <- 
│  ├── base.py          <- 
│  ├── inference_core.py        <- 
│  ├── interactive_utils.py     <- 
│  ├── interface.py             <- 
│  ├── kv_memory_store.py       <- 
│  └── memory_manager.py/       <- 
└── tests/     <-  功能测试目录
```

## **3.环境与依赖**

* python3.8+
* pillow
* torchvision
* numpy

## **4. 安装说明**

1. 安装Python 3.8或者更高版本
2. 克隆项目到本地：`git clone http://192.168.0.188:8090/ai_lab_rd02/ai_sdks/xmem.git`
3. 进入项目目录：`cd xmem`
4. 安装依赖：`pip install -r requirements.txt`
5. 编译打包：在与 `setup.py `文件相同的目录下执行以下命令：`python setup.py bdist_wheel`。 在 `dist` 文件夹中找到 `.wheel`
   文件，例如：`dist/xmem-0.1.0-py3-none-any.whl`。
6. 安装：`pip install xmem-0.1.0-py3-none-any.whl`

## **5. 使用指南**

## 6. 接口示例

```python
"""
You can find the testing data in folder: tests
"""
from rmx.interface import TrackRmx
import numpy as np
from PIL import Image
from rmx.util import palette  # 导入自定义的调色板

track = TrackRmx()
processor = track.gen_model("tests/XMem.pth")

mark_image = Image.open("1.jpg")  # first frame
mark_image = np.array(mark_image)

mark_mask = Image.open("1.png")  # mask of first frame
mark_mask = np.array(mark_mask)

predict_image = Image.open("2.jpg")
predict_image = np.array(predict_image)

num_objects = 1

predict_mask = track.detect(processor, mark_image, mark_mask, predict_image, num_objects)
print(predict_mask)

predict_mask = Image.fromarray(predict_mask)
predict_mask.putpalette(palette.davis_palette)
predict_mask.show()

track.delete_model(processor)
```

## 7. **许可证信息**

说明项目的开源许可证类型（如MIT、Apache 2.0等）。

* 本项目遵循MIT许可证。

## 8. 常见问题解答（FAQ）**

列出一些常见问题和解决方案。

- **Q1：机械臂连接失败**

  答案：修改过机械臂IP

- **Q2：UDP数据推送接口收不到数据**

  答案：检查线程模式、是否使能推送数据、IP以及防火墙