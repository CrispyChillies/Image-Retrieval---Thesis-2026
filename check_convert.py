import onnxruntime as ort
p="/home/tuktu/Image-Retrieval---Thesis-2026/ChestMIR/weights/stage1_fold3.onnx"
s=ort.InferenceSession(p, providers=["CPUExecutionProvider"])
print("input:", s.get_inputs()[0].shape)
print("output:", s.get_outputs()[0].shape)