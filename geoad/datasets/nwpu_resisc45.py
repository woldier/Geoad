# -*- coding:utf-8 -*-
"""
 @FileName   : nwpu_resisc45.py
 @Time       : 2025/7/3 14:00
 @Author     : Woldier Wong
 @Description: nwpu_resisc45
"""

from typing import List, Optional, Union

from mmengine import fileio

from mmpretrain.registry import DATASETS
from .categories import RESISC45_CATEGORIES
from mmpretrain.datasets.custom import CustomDataset

@DATASETS.register_module()
class RESISC45(CustomDataset):
    """
        RESISC45: Remote Sensing Image Scene Classification: Benchmark and State of the Art
        https://gcheng-nwpu.github.io/#Datasets
        Args:
            data_root (str): The root directory for ``data_prefix`` and
                ``ann_file``. Defaults to ''.
            data_prefix (str | dict): Prefix for training data. Defaults to ''.
            ann_file (str): Annotation file path. Defaults to ''.
            metainfo (dict, optional): Meta information for dataset, such as class
                information. Defaults to None.
            **kwargs: Other keyword arguments in :class:`CustomDataset` and
                :class:`BaseDataset`.
        Examples:
            >>> from PeftPretrain.datasets.nwpu_resisc45 import RESISC45
            >>> dataset  = RESISC45(data_root='./data/NWPU-RESISC45',ann_file='ratio_0.5_train.txt')
            >>> data = dataset[0]
            >>> print(dataset)
            Dataset RESISC45
                Number of samples: 	15750
                Number of categories: 	45
                Root of dataset: 	./data/NWPU-RESISC45
        """  # noqa
    METAINFO = {'classes': RESISC45_CATEGORIES}

    def __init__(self,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        _ann_path = fileio.join_path(data_root, ann_file)
        assert fileio.exists(_ann_path), f'ann_file {_ann_path} not exist!'
        super().__init__(data_root=data_root,
                         data_prefix=data_prefix,
                         ann_file=ann_file,
                         metainfo=metainfo,
                         **kwargs)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body