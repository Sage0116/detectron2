from detectron2.utils import comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation import verify_results

import os
import sys
from datetime import datetime
sys.path.append(os.getcwd())

from detectron2.modeling.trainer import DATrainer,  DefaultTrainer_

# register datasets
import detectron2.data.register

# register compoments
from detectron2.modeling.meta_arch.sw import SWDARCNN
from detectron2.modeling.backbone.dafpn import build_resnet_fpn_da_backbone
from detectron2.modeling.da_heads.da_head import AlignmentHead
from detectron2.modeling.da_heads.da_roi_head import DAROIHeads


def add_swdarcnn_config(cfg):
    from detectron2.config import CfgNode as CN
    _C = cfg
    _C.MODEL.DA_HEADS = CN()
    _C.MODEL.DOMAIN_ADAPTATION_ON = True
    _C.MODEL.DA_HEADS.LOCAL_ALIGNMENT_ON = True
    _C.MODEL.DA_HEADS.GLOBAL_ALIGNMENT_ON = True
    _C.MODEL.DA_HEADS.GAMMA = 5.0
    _C.MODEL.ROI_HEADS.CONTEXT_REGULARIZATION_ON = True
    _C.MODEL.ROI_HEADS.CONTEXT_REGULARIZATION_FEATURES = ['local_head_feature', 'global_head_feature']
    _C.DATASETS.SOURCE_DOMAIN = CN()
    _C.DATASETS.SOURCE_DOMAIN.TRAIN = ()
    _C.DATASETS.TARGET_DOMAIN = CN()
    _C.DATASETS.TARGET_DOMAIN.TRAIN = ()


def setup(args):
    cfg = get_cfg()
    add_swdarcnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    now = datetime.now()
    if not args.resume:
        cfg.OUTPUT_DIR = './outputs/output-{}'.format(now.strftime("%y-%m-%d_%H-%M"))
        if args.setting_token:
            cfg.OUTPUT_DIR = './outputs/output-{}-{}'.format(args.setting_token, now.strftime("%y-%m-%d_%H-%M"))
    cfg.freeze()
    assert cfg.MODEL.DOMAIN_ADAPTATION_ON or not cfg.MODEL.ROI_HEADS.CONTEXT_REGULARIZATION_ON, \
        'when using context regularization, network must have domain adapatation head'
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        assert cfg.MODEL.DA_HEADS.LOCAL_ALIGNMENT_ON or cfg.MODEL.DA_HEADS.GLOBAL_ALIGNMENT_ON, \
        'domain adapatation head must have one alignment head (local or global) at least'
    if not args.test_images:
        default_setup(cfg, args)
    return cfg

def test_images(cfg):
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import load_voc_instances
    import cv2
    from pathlib import Path
    predictor = DefaultPredictor(cfg)

    for dataset_name in cfg.DATASETS.TEST:
        now = datetime.now()
        output_dir = Path(__file__).parent.parent/ 'test_images'/ (dataset_name + '-' + now.strftime("%y-%m-%d_%H-%M"))
        if not output_dir.parent.is_dir():
            output_dir.parent.mkdir()
        if not output_dir.is_dir():
            output_dir.mkdir()
        dirname = MetadataCatalog.get(dataset_name).get('dirname')
        split = MetadataCatalog.get(dataset_name).get('split')
        thing_classes = MetadataCatalog.get(dataset_name).get('thing_classes')
        for d in iter(load_voc_instances(dirname, split, thing_classes)):
            im = cv2.imread(d.get('file_name'))
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(dataset_name), scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            outputs = predictor(im)
            cv2.imwrite(str(output_dir/'{}.jpg').format(Path(d.get('file_name')).stem), v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1])

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = DATrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = DATrainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    if args.test_images:
        test_images(cfg)
        return

    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        trainer = DATrainer(cfg)
    else:
        trainer = DefaultTrainer_(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--test-images", action="store_true", help="output predicted bbox to test images")
    parser.add_argument("--setting-token", help="add some simple profile about this experiment to output directory name")
    args = parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )