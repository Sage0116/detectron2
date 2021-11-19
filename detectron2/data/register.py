from detectron2.data.datasets import register_pascal_voc
from pathlib import Path

dataset_base_dir = Path(__file__).parent.parent.parent.parent / 'data'



dataset_dir = str(dataset_base_dir/ 'VOCdevkit')
classes = ("bicycle", "bird", "car", "cat", "dog", "person")
years = 2007
split = 'trainval' # "train", "test", "val", "trainval"
meta_name = 'pascal_trainval_2007'
register_pascal_voc(meta_name, dataset_dir+'/VOC2007/', split, years, classes)
years = 2012
split = 'trainval' # "train", "test", "val", "trainval"
meta_name = 'pascal_trainval_2012'
register_pascal_voc(meta_name, dataset_dir+'/VOC2012/', split, years, classes)


dataset_dir = str(dataset_base_dir/ 'watercolor')
classes = ("bicycle", "bird", "car", "cat", "dog", "person")
years = 2007
split = 'trainval' # "train", "test", "val", "trainval"
meta_name = 'watercolor_trainval'
register_pascal_voc(meta_name, dataset_dir+'/VOC2007/', split, years, classes)



dataset_dir = str(dataset_base_dir/ 'watercolor')
split = 'test' # "train", "test", "val", "trainval"
classes = ("bicycle", "bird", "car", "cat", "dog", "person")
years = 2007
meta_name = 'watercolor_test'
register_pascal_voc(meta_name, dataset_dir+'/VOC2007/', split, years, classes)

dataset_dir = str(dataset_base_dir/ 'cityscape')
classes = ('bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
years = 2007
split = 'trainval' # "train", "test", "val", "trainval"
meta_name = 'cityscape_trainval'
register_pascal_voc(meta_name, dataset_dir+'/VOC2007/', split, years, classes)

dataset_dir = str(dataset_base_dir/ 'foggycity')
classes = ('bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
years = 2007
split = 'trainval' # "train", "test", "val", "trainval"
meta_name = 'foggycity_trainval'
register_pascal_voc(meta_name, dataset_dir+'/VOC2007/', split, years, classes)

dataset_dir = str(dataset_base_dir/ 'foggycity')
classes = ('bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
years = 2007
split = 'test' # "train", "test", "val", "trainval"
meta_name = 'foggycity_test'
register_pascal_voc(meta_name, dataset_dir+'/VOC2007/', split, years, classes)