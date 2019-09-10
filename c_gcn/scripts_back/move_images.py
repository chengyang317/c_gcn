from pathlib import Path

import shutil

img_dir = Path('/home/philip/sdb/datasets/coco/coco_train2014')

dest_dir = Path('/home/philip/sdb/datasets/coco')

for i in range(500, 2000):
    img_file = img_dir.joinpath('COCO_train2014_000000{:0>6}.jpg'.format(i))
    if img_file.exists():
        shutil.copy(img_file, dest_dir)





