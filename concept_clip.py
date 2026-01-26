from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image

model = AutoModel.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('JerrryNie/ConceptCLIP', trust_remote_code=True)

image = Image.open('example_data/chest_X-ray.jpg').convert('RGB')
labels = ['chest X-ray', 'brain MRI', 'skin lesion']
texts = [f'a medical image of {label}' for label in labels]

inputs = processor(
    images=image, 
    text=texts,
    return_tensors='pt',
    padding=True,
    truncation=True
).to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = (outputs['logit_scale'] * outputs['image_features'] @ outputs['text_features'].t()).softmax(dim=-1)[0]

print({label: f"{prob:.2%}" for label, prob in zip(labels, logits)})
