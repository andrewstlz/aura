"""
Background Removal using U²-Net P Model
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class REBNCONV(torch.nn.Module):
    # Residual block with batch normalization and ReLU activation
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = torch.nn.BatchNorm2d(out_ch)
        self.relu_s1 = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU7(torch.nn.Module):
    # Residual U-block with 7 layers
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 1)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, 1)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, 1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, 2)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, 1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


class RSU6(torch.nn.Module):
    # Residual U-block with 6 layers
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 1)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, 1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, 2)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, 1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


class RSU5(torch.nn.Module):
    # Residual U-block with 5 layers
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, 2)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, 1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


class RSU4(torch.nn.Module):
    # Residual U-block with 4 layers
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 2)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, 1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


class RSU4F(torch.nn.Module):
    # Residual U-block with 4 layers and dilated convolutions
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 8)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, 4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, 2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, 1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hxin


class U2NETP(torch.nn.Module):
    # U²-Net Portrait model for salient object detection
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()
        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(64, 16, 64)
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)
        self.side1 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.outconv = torch.nn.Conv2d(6*out_ch, out_ch, 1)

    def forward(self, x):
        hx = x
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d3 = self.side3(hx3d)
        d4 = self.side4(hx4d)
        d5 = self.side5(hx5d)
        d6 = self.side6(hx6)
        d2 = F.interpolate(d2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(d3, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d4 = F.interpolate(d4, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d5 = F.interpolate(d5, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d6 = F.interpolate(d6, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)


def load_model(model_path):
    # Load pretrained U²-Net P model weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print("Loading U²-Net P model...")
    model = U2NETP(3, 1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def process_image(model, input_path, output_path):
    # Generate transparent background image with alpha mask
    image = Image.open(input_path).convert('RGB')
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        d1, _, _, _, _, _, _ = model(input_tensor)
    
    pred = d1[:, 0, :, :]
    pred = F.interpolate(pred.unsqueeze(1), size=original_size[::-1], mode='bilinear', align_corners=False)
    mask = pred.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    
    mask_pil = Image.fromarray(mask).convert('L')
    r, g, b = image.split()
    rgba = Image.merge('RGBA', (r, g, b, mask_pil))
    
    if not output_path.lower().endswith('.png'):
        output_path = os.path.splitext(output_path)[0] + '.png'
    
    rgba.save(output_path)
    print(f"Saved transparent output to: {output_path}")


if __name__ == "__main__":
    model = load_model("/Users/Zheng/Desktop/CIS5810Project/u2netp.pth")
    
    input_image = "/Users/zheng/Downloads/Shi-2631-1.jpg"
    output_image = "/Users/zheng/Desktop/CIS5810Project/Shi-output.png"
    
    process_image(model, input_image, output_image)