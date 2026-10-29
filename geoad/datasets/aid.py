# -*- coding:utf-8 -*-
"""
 @FileName   : aid.py
 @Time       : 4/16/25 9:25 PM
 @Author     : Woldier Wong
 @Description: AID dataset
"""

from typing import List, Optional, Union

from mmengine import fileio

from mmpretrain.registry import DATASETS
from .categories import AID_CATEGORIES
from mmpretrain.datasets.custom import CustomDataset


@DATASETS.register_module()
class AID(CustomDataset):
    """
    AID: A Benchmark Dataset for Performance Evaluation of Aerial Scene Classification
    https://captain-whu.github.io/AID/
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
        >>> from mmpretrain.datasets.aid import AID
        >>> dataset  = AID(data_root='./data/aid',ann_file='ratio_0.5_train.txt')
        >>> data = dataset[0]
        >>> print(dataset)
        Dataset AID
        Number of samples: 	5000
        Number of categories: 	30
        Root of dataset: 	./data/aid
    """ # noqa
    METAINFO = {'classes': AID_CATEGORIES}

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
