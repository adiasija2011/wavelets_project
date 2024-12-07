import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Wavelet Transform functions
class WaveletTransform(nn.Module):
    def __init__(self):
        super(WaveletTransform, self).__init__()
    
    def forward(self, batch_image):
        # batch_image shape: (batch_size, channels, height, width)
        # Assuming channels = 3

        # Split into R, G, B channels
        r = batch_image[:, 0, :, :]
        g = batch_image[:, 1, :, :]
        b = batch_image[:, 2, :, :]

        # Level 1 decomposition
        r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH = self.wavelet_decompose(r)
        g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH = self.wavelet_decompose(g)
        b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH = self.wavelet_decompose(b)

        wavelet_data = [r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH, 
                        g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH,
                        b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH]
        transform_batch = torch.stack(wavelet_data, dim=1)  # shape: (batch_size, 12, h, w)

        # Level 2 decomposition
        r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2 = self.wavelet_decompose(r_wavelet_LL)
        g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2 = self.wavelet_decompose(g_wavelet_LL)
        b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2 = self.wavelet_decompose(b_wavelet_LL)

        wavelet_data_l2 = [r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2, 
                           g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2,
                           b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2]
        transform_batch_l2 = torch.stack(wavelet_data_l2, dim=1)

        # Level 3 decomposition
        r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3 = self.wavelet_decompose(r_wavelet_LL2)
        g_wavelet_LL3, g_wavelet_LH3, g_wavelet_HL3, g_wavelet_HH3 = self.wavelet_decompose(g_wavelet_LL2)
        b_wavelet_LL3, b_wavelet_LH3, b_wavelet_HL3, b_wavelet_HH3 = self.wavelet_decompose(b_wavelet_LL2)

        wavelet_data_l3 = [r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3, 
                           g_wavelet_LL3, g_wavelet_LH3, g_wavelet_HL3, g_wavelet_HH3,
                           b_wavelet_LL3, b_wavelet_LH3, b_wavelet_HL3, b_wavelet_HH3]
        transform_batch_l3 = torch.stack(wavelet_data_l3, dim=1)

        # Level 4 decomposition
        r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4 = self.wavelet_decompose(r_wavelet_LL3)
        g_wavelet_LL4, g_wavelet_LH4, g_wavelet_HL4, g_wavelet_HH4 = self.wavelet_decompose(g_wavelet_LL3)
        b_wavelet_LL4, b_wavelet_LH4, b_wavelet_HL4, b_wavelet_HH4 = self.wavelet_decompose(b_wavelet_LL3)

        wavelet_data_l4 = [r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4, 
                           g_wavelet_LL4, g_wavelet_LH4, g_wavelet_HL4, g_wavelet_HH4,
                           b_wavelet_LL4, b_wavelet_LH4, b_wavelet_HL4, b_wavelet_HH4]
        transform_batch_l4 = torch.stack(wavelet_data_l4, dim=1)

        return [transform_batch, transform_batch_l2, transform_batch_l3, transform_batch_l4]
    
    def wavelet_decompose(self, channel):
        # channel shape: (batch_size, h, w)
        wavelet_L, wavelet_H = self.WaveletTransformAxisY(channel)
        wavelet_LL, wavelet_LH = self.WaveletTransformAxisX(wavelet_L)
        wavelet_HL, wavelet_HH = self.WaveletTransformAxisX(wavelet_H)
        return wavelet_LL, wavelet_LH, wavelet_HL, wavelet_HH

    def WaveletTransformAxisY(self, batch_img):
        # batch_img shape: (batch_size, h, w)
        odd_img  = batch_img[:, 0::2, :]
        even_img = batch_img[:, 1::2, :]
        L = (odd_img + even_img) / 2.0
        H = torch.abs(odd_img - even_img)
        return L, H

    def WaveletTransformAxisX(self, batch_img):
        # batch_img shape: (batch_size, h, w)
        # transpose + flip left-right
        tmp_batch = torch.flip(batch_img.transpose(1, 2), [2])
        dst_L, dst_H = self.WaveletTransformAxisY(tmp_batch)
        # transpose + flip up-down
        dst_L = torch.flip(dst_L.transpose(1, 2), [1])
        dst_H = torch.flip(dst_H.transpose(1, 2), [1])
        return dst_L, dst_H
    
# Define the model architecture
class WaveletCNNModel(nn.Module):
    def __init__(self, num_classes=12):
        super(WaveletCNNModel, self).__init__()
        self.wavelet = WaveletTransform()
        # Level 1
        self.conv_1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.norm_1 = nn.BatchNorm2d(64)
        
        self.conv_1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.norm_1_2 = nn.BatchNorm2d(64)
        
        # Level 2
        self.conv_a = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.norm_a = nn.BatchNorm2d(64)
        
        # After concat level 2
        self.conv_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm_2 = nn.BatchNorm2d(128)
        
        self.conv_2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.norm_2_2 = nn.BatchNorm2d(128)
        
        # Level 3
        self.conv_b = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.norm_b = nn.BatchNorm2d(64)
        
        self.conv_b_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm_b_2 = nn.BatchNorm2d(128)
        
        # After concat level 3
        self.conv_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.norm_3 = nn.BatchNorm2d(256)
        
        self.conv_3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.norm_3_2 = nn.BatchNorm2d(256)
        
        # Level 4
        self.conv_c = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.norm_c = nn.BatchNorm2d(64)
        
        self.conv_c_2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.norm_c_2 = nn.BatchNorm2d(256)
        
        self.conv_c_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.norm_c_3 = nn.BatchNorm2d(256)
        
        # After concat level 4
        self.conv_4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.norm_4 = nn.BatchNorm2d(256)
        
        self.conv_4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.norm_4_2 = nn.BatchNorm2d(256)
        
        # Final layers
        self.conv_5_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.norm_5_1 = nn.BatchNorm2d(128)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        
        self.fc_5 = nn.Linear(128 * 7 * 7, 1024)
        self.norm_5 = nn.BatchNorm1d(1024)
        self.drop_5 = nn.Dropout(0.5)
        
        self.output = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224)
        input_l1, input_l2, input_l3, input_l4 = self.wavelet(x)
        # Level 1
        x1 = F.relu(self.norm_1(self.conv_1(input_l1)))
        x1 = F.relu(self.norm_1_2(self.conv_1_2(x1)))
        
        # Level 2
        x2 = F.relu(self.norm_a(self.conv_a(input_l2)))
        
        # Concatenate level 1 and 2
        x12 = torch.cat([x1, x2], dim=1)  # dim=1 is the channel dimension
        x12 = F.relu(self.norm_2(self.conv_2(x12)))
        x12 = F.relu(self.norm_2_2(self.conv_2_2(x12)))
        
        # Level 3
        x3 = F.relu(self.norm_b(self.conv_b(input_l3)))
        x3 = F.relu(self.norm_b_2(self.conv_b_2(x3)))
        
        # Concatenate level 2 and 3
        x123 = torch.cat([x12, x3], dim=1)
        x123 = F.relu(self.norm_3(self.conv_3(x123)))
        x123 = F.relu(self.norm_3_2(self.conv_3_2(x123)))
        
        # Level 4
        x4 = F.relu(self.norm_c(self.conv_c(input_l4)))
        x4 = F.relu(self.norm_c_2(self.conv_c_2(x4)))
        x4 = F.relu(self.norm_c_3(self.conv_c_3(x4)))
        
        # Concatenate level 3 and 4
        x1234 = torch.cat([x123, x4], dim=1)
        x1234 = F.relu(self.norm_4(self.conv_4(x1234)))
        x1234 = F.relu(self.norm_4_2(self.conv_4_2(x1234)))
        
        x5 = F.relu(self.norm_5_1(self.conv_5_1(x1234)))
        x5 = self.avg_pool(x5)  # Output shape: (batch_size, 128, 7, 7)
        x5 = x5.view(x5.size(0), -1)  # Flatten
        x5 = F.relu(self.norm_5(self.fc_5(x5)))
        x5 = self.drop_5(x5)
        output = self.output(x5)
        return output