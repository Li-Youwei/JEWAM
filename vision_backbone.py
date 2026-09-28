"""Legacy checkpoint import path; use :mod:`jewam.models.vision` in new code."""

from jewam.models.vision import (
    HFVisionBackbone as HFVisionBackbone,
)
from jewam.models.vision import (
    build_visual_encoder as build_visual_encoder,
)
