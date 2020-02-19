from torchvision import models
import torch
 
dir(models)

from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

from PIL import Image
img = Image.open("envelope.jpeg")

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

resnet = models.resnet101(pretrained=True)
 
resnet.eval()
 
out = resnet(batch_t)

with open('imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]
 
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
[(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
for idx in indices[0][:2]:
    print(classes[idx] + "")
