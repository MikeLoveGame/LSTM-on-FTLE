import torch
import torch.nn.functional as F
import torch.nn as nn

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2, channel=9, device = "cuda:0"):
        super(SSIMLoss, self).__init__()
        self.device = device
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        self.channel = channel
        self.window = self.create_window(window_size, channel)
        self.to(device=device)


    def forward(self, img1, img2):
        print(img1.device)
        print(img2.device)
        print(self.window.device)
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        return 1 - ssim_map.mean()

    def create_window(self, window_size, channel):
        window = torch.zeros(window_size, window_size, device = self.device)
        center = window_size // 2
        sigma = 1.5

        def gauss(x, y):
            input = -(x - center) ** 2 / (2 * sigma ** 2) - (y - center) ** 2 / (2 * sigma ** 2)
            return torch.exp(torch.as_tensor(input))

        for x in range(0, window_size):
            for y in range(0, window_size):
                window[x, y] = gauss(x, y)

        # Normalize the window
        window /= window.sum()

        window = window.view(1, 1, window_size, window_size)
        window = window.repeat(channel, 1, 1, 1)
        return window

# Test
def main():
    ssim_loss = SSIMLoss()
    img1 = torch.randn(1, 2, 128, 128)
    img2 = torch.randn(1, 2, 128, 128)
    loss = ssim_loss(img1, img2)

    print(f"SSIM Loss: {loss.item()}")
