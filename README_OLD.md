## Introduction
该模型可以根据目标物体在第一帧的掩码(mask)描述，在后续帧中跟踪该物体并生成相应的掩码(mask)

## Prequests
首先需要根据本机的cuda driver版本和cuda版本来安装正确的pytorch

## Weights address
https://alidocs.dingtalk.com/i/nodes/oP0MALyR8kRvgalNI3z4O5vKJ3bzYmDO?utm_scene=team_space

## Use Case
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

mark_image = Image.open("1.jpg") # first frame
mark_image = np.array(mark_image)

mark_mask = Image.open("1.png") # mask of first frame
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

## Distribute
Execute the following commands in the same directory as setup.py: `python setup.py bdist_wheel`.
find the `.wheel` in folder dist, example: `dist/vertical_sam-0.1.0-py3-none-any.whl`

## Install
`pip install <package-name>.whl`

## Unit Test
1. cd into tests 
2. execute: `pytest` 