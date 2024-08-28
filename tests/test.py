import sys
sys.path.append("..\src")

from interface import TrackRmx
import numpy as np
from PIL import Image
from util import palette  # 导入自定义的调色板


track = TrackRmx()
processor = track.gen_model("./XMem.pth")

mark_image = Image.open("1.jpg")
mark_image = np.array(mark_image)

mark_mask = Image.open("1.png")
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