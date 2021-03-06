import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from fvcore.nn.focal_loss import sigmoid_focal_loss_jit
from .build import DA_HEAD_REGISTRY
from detectron2.layers.gradient_scalar_layer import GradientScalarLayer

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

class GlobalAlignmentHead(nn.Module):
  def __init__(self, context=False):
    super(GlobalAlignmentHead, self).__init__()
    self.conv1 = conv3x3(1024, 512, stride=2)
    self.bn1 = nn.BatchNorm2d(512)
    self.conv2 = conv3x3(512, 128, stride=2)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = conv3x3(128, 128, stride=2)
    self.bn3 = nn.BatchNorm2d(128)
    self.fc = nn.Linear(128, 2)
    self.context = context
    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
    x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
    x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
    x = F.avg_pool2d(x, (x.size(2),x.size(3)))
    x = x.view(-1, 128)
    if self.context:
      feat = x
    x = self.fc(x)
    if self.context:
      return x, feat
    else:
      return x


class LocalAlignmentHead(nn.Module):
  def __init__(self, context=False):
    super(LocalAlignmentHead, self).__init__()
    self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
    self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
    self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)
    self.context = context
    self._init_weights()

  def _init_weights(self):
    def normal_init(m, mean, stddev,truncated=False):
      """
      weight initalizer: random normal.
      """
      if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) 
      else:
          m.weight.data.normal_(mean, stddev)
    

    normal_init(self.conv1, 0, 0.01)
    normal_init(self.conv2, 0, 0.01)
    normal_init(self.conv3, 0, 0.01)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    if self.context:
      feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
      x = self.conv3(x)
      return  F.sigmoid(x), feat
    else:
      x = self.conv3(x)
      return F.sigmoid(x)


@DA_HEAD_REGISTRY.register()
class AlignmentHead(nn.Module):

  @configurable
  def __init__(self, *, local_alignment_on, global_alignment_on, gamma=5.0):
    super().__init__()
    if local_alignment_on:
      self.localhead = LocalAlignmentHead(context=True)
      self.grl_localhead = GradientScalarLayer(-1.0)
    else:
      self.localhead = None
    if global_alignment_on:
      self.globalhead = GlobalAlignmentHead(context=True)
      self.grl_globalhead = GradientScalarLayer(-1.0)
    else:
      self.globalhead = None
    self.gamma = gamma

  @classmethod
  def from_config(cls, cfg):
    assert cfg.MODEL.DA_HEADS.LOCAL_ALIGNMENT_ON or cfg.MODEL.DA_HEADS.GLOBAL_ALIGNMENT_ON, 'domain adapatation head must have one alignment head at least'
    return {
      'gamma': cfg.MODEL.DA_HEADS.GAMMA,
      'local_alignment_on': cfg.MODEL.DA_HEADS.LOCAL_ALIGNMENT_ON,
      'global_alignment_on': cfg.MODEL.DA_HEADS.GLOBAL_ALIGNMENT_ON,
    }
  
  def forward(self, inputs):

    feat_local = inputs['local_head_feature']
    feat_global = inputs['global_head_feature']
    feat_domain = inputs['feature domain']

    reg_local_feat = None
    reg_global_feat = None
    loss = {}

    if self.localhead:
      _, reg_local_feat = self.localhead(feat_local.detach())
    if self.globalhead:
      _, reg_global_feat = self.globalhead(feat_global.detach())

    if self.training:
      if self.localhead:
        feat_2d, _ = self.localhead(self.grl_localhead(feat_local))
        # local alignment loss
        if feat_domain == 'source':
          domain_loss_local = torch.mean(feat_2d ** 2)
        elif feat_domain =='target':
          domain_loss_local = torch.mean((1-feat_2d) ** 2)
        loss.update({'loss_local_alignment': domain_loss_local})

      if self.globalhead:
        feat_value, _ = self.globalhead(self.grl_globalhead(feat_global))
        if feat_domain == 'source':
          domain_label = torch.zeros_like(feat_value, requires_grad=True, device=feat_value.device)
        elif feat_domain == 'target':
          domain_label = torch.ones_like(feat_value, requires_grad=True, device=feat_value.device)
        # global alignment loss
        focal_loss_global = sigmoid_focal_loss_jit(feat_value, domain_label, gamma=self.gamma, reduction='mean')
        loss.update({'loss_global_alignment': focal_loss_global})

      return {'local_head_feature': reg_local_feat, 'global_head_feature': reg_global_feat}, loss

    else:
      return {'local_head_feature': reg_local_feat, 'global_head_feature': reg_global_feat}


def build_da_heads(cfg):
    return AlignmentHead(cfg)