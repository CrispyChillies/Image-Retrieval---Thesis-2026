from pathlib import Path
import torch
from model import ConvNeXtV2

ckpt_path = Path("/home/tuktu/Image-Retrieval---Thesis-2026/covid_convnextv2_seed_0_epoch_16_ckpt.pth")  # <-- sửa
out_path = Path("/home/tuktu/Image-Retrieval---Thesis-2026/ChestMIR/weights/retrieval_model/covid_convnextv2_seed_0_epoch_16_backbone.onnx")

model = ConvNeXtV2(embedding_dim=None)
ckpt = torch.load(ckpt_path, map_location="cpu")
if isinstance(ckpt, dict) and "state-dict" in ckpt:
    ckpt = ckpt["state-dict"]
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]
model.load_state_dict(ckpt, strict=False)
model.eval()

dummy = torch.randn(1, 3, 384, 384, dtype=torch.float32)
torch.onnx.export(
    model,
    dummy,
    str(out_path),
    input_names=["input"],
    output_names=["features"],
    dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
    opset_version=17,
)
print("Exported:", out_path)