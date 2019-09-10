# coding=utf-8
from torchvision import transforms
import tqdm
import h5py
from pt_pack.datasets.reader import ImgDirReader, H5Reader
from pt_pack.utils import to_path


__all__ = ['ClevrImgReader', 'ClevrImgH5Reader']


class ClevrImgReader(ImgDirReader):
    def __init__(self,
                 data_dir,
                 split,
                 req_field_names=None,
                 transforms=None,
                 is_load=False,
                 ):
        self.data_dir = to_path(data_dir)
        if split in ('valB_30k', 'valB_120k'):
            split = 'valB'
        super().__init__(self.data_dir.parent.joinpath('images', f'{split}'), req_field_names, transforms=transforms)

    @staticmethod
    def load_transforms(img_size):
        tfs = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Pad(4),
            transforms.RandomCrop([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return tfs


class ClevrImgH5Reader(H5Reader):
    valid_field_names = ('images', 'img_ids')

    def __init__(self,
                 data_dir,
                 split,
                 req_field_names=None,
                 is_load=False,
                 ):
        self.data_dir = to_path(data_dir)
        h5_file = self.data_dir.joinpath(f'{split}_images.h5')
        if not h5_file.exists():
            self.create_h5_file(self.data_dir, split)
        super().__init__(h5_file, req_field_names, is_load=is_load)

    def create_h5_file(self, data_dir, split):
        h5_file = self.data_dir.joinpath(f'{split}_images.h5')
        print(f'Creating {h5_file.name}...')
        h5_fd = h5py.File(h5_file, 'w')

        tfs = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        reader = ClevrImgReader(self.data_dir, split, transforms=tfs)
        dset_len = len(reader['img_ids'])

        d_sets = dict()
        for dset_idx in tqdm.tqdm(range(dset_len)):
            if 'img_ids' not in d_sets:
                d_sets['img_ids'] = h5_fd.create_dataset('img_ids', (dset_len,))
            d_sets['img_ids'][dset_idx] = reader['img_ids'][dset_idx]

            image = reader['images'][dset_idx].numpy()
            if 'images' not in d_sets:
                d_sets['images'] = h5_fd.create_dataset('images', (dset_len, *image.shape))
            d_sets['images'][dset_idx] = image
        h5_fd.close()
