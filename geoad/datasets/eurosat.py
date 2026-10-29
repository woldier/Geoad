# -*- coding:utf-8 -*-
"""
 @FileName   : eurosat.py
 @Time       : 2025/7/3 14:00
 @Author     : Woldier Wong
 @Description: EuroSat
"""

from typing import List, Optional, Union

from mmengine import fileio

from mmpretrain.registry import DATASETS
from .categories import EUROSAT_CATEGORIES
from mmpretrain.datasets.custom import CustomDataset

@DATASETS.register_module()
class EuroSAT(CustomDataset):
    """
        EuroSAT: Land Use and Land Cover Classification with Sentinel-2
        https://github.com/phelber/eurosat
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
            >>> from PeftPretrain.datasets.eurosat import EuroSAT
            >>> dataset  = EuroSAT(data_root='./data/EuroSAT_RGB', ann_file='eurosat-train.txt')
            >>> print(dataset)
            Number of samples: 	16200
            Number of categories: 	10
            Root of dataset: 	/home/woldier/data/EuroSAT_RGB
        """  # noqa
    METAINFO = {'classes': EUROSAT_CATEGORIES}

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