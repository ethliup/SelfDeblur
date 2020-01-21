import cv2
import torch
import numpy as np
import torch.nn as nn

def white_balance(img):
    img = (img*255.).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img[:, :, 1])
    avg_b = np.average(img[:, :, 2])
    img[:, :, 1] = img[:, :, 1] - ((avg_a - 128) * (img[:, :, 0] / 255.0) * 1.1)
    img[:, :, 2] = img[:, :, 2] - ((avg_b - 128) * (img[:, :, 0] / 255.0) * 1.1)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img = img.astype(np.float)/255.
    return img

def warp_image_flow(ref_image, flow):
    [B, _, H, W] = ref_image.size()
    
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if ref_image.is_cuda:
        grid = grid.cuda()

    flow_f = flow + grid
    flow_fx = flow_f[:, 0, :, :] 
    flow_fy = flow_f[:, 1, :, :]

    with torch.no_grad():
        mask_x = ~((flow_fx < 0) | (flow_fx > (W - 1)))
        mask_y = ~((flow_fy < 0) | (flow_fy > (H - 1)))
        mask = mask_x & mask_y
        mask = mask.unsqueeze(1)

    flow_fx = flow_fx / float(W) * 2. - 1.
    flow_fy = flow_fy / float(H) * 2. - 1.

    flow_fxy = torch.stack([flow_fx, flow_fy], dim=-1)
    img = torch.nn.functional.grid_sample(ref_image, flow_fxy, padding_mode='zeros') 
    return img, mask

class Grid_gradient_central_diff():
    def __init__(self, nc, padding=True, diagonal=False):
        self.conv_x = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_y = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_xy = None
        if diagonal:
            self.conv_xy = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
    
        self.padding=None
        if padding:
            self.padding = nn.ReplicationPad2d([0,1,0,1])

        fx = torch.zeros(nc, nc, 2, 2).float().cuda()
        fy = torch.zeros(nc, nc, 2, 2).float().cuda()
        if diagonal:
            fxy = torch.zeros(nc, nc, 2, 2).float().cuda()
        
        fx_ = torch.tensor([[1,-1],[0,0]]).cuda()
        fy_ = torch.tensor([[1,0],[-1,0]]).cuda()
        if diagonal:
            fxy_ = torch.tensor([[1,0],[0,-1]]).cuda()

        for i in range(nc):
            fx[i, i, :, :] = fx_
            fy[i, i, :, :] = fy_
            if diagonal:
                fxy[i,i,:,:] = fxy_
            
        self.conv_x.weight = nn.Parameter(fx)
        self.conv_y.weight = nn.Parameter(fy)
        if diagonal:
            self.conv_xy.weight = nn.Parameter(fxy)

    def __call__(self, grid_2d):
        _image = grid_2d
        if self.padding is not None:
            _image = self.padding(_image)
        dx = self.conv_x(_image)
        dy = self.conv_y(_image)

        if self.conv_xy is not None:
            dxy = self.conv_xy(_image)
            return dx, dy, dxy
        return dx, dy