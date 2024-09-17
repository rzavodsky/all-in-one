import torch
import torch.nn as nn

from typing import Callable, List
from .allinone import AllInOne
from ..typings import AllInOneOutput


class Ensemble(nn.Module):
  def __init__(self, models: List[AllInOne]):
    super().__init__()

    cfg = models[0].cfg.copy()
    cfg.best_threshold_beat = sum([model.cfg.best_threshold_beat for model in models]) / len(models)
    cfg.best_threshold_downbeat = sum([model.cfg.best_threshold_downbeat for model in models]) / len(models)

    self.cfg = cfg
    self.models = models

  def forward(self, x, *, progress_callback: Callable[[float], None] | None = None):
    outputs: List[AllInOneOutput] = []
    for i, model in enumerate(self.models):
      outputs.append(model(
        x,
        progress_callback=(lambda x: progress_callback((i+x) / len(self.models))) if progress_callback else None,
      ))
    avg = AllInOneOutput(
      logits_beat=torch.stack([output.logits_beat for output in outputs], dim=0).mean(dim=0),
      logits_downbeat=torch.stack([output.logits_downbeat for output in outputs], dim=0).mean(dim=0),
      logits_section=torch.stack([output.logits_section for output in outputs], dim=0).mean(dim=0),
      logits_function=torch.stack([output.logits_function for output in outputs], dim=0).mean(dim=0),
      embeddings=torch.stack([output.embeddings for output in outputs], dim=-1),
    )

    return avg
