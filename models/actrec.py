import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import curves
from torchsummary import summary

class ActRecBase(nn.Module):
    def __init__(self, num_classes):
        super(ActRecBase, self).__init__()
        self.conv1 = nn.Conv1d(6, 256, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 =  nn.Dropout(0)
        self.relu2 = nn.ReLU(True)
        
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.dropout3 =  nn.Dropout(0)
        self.relu3 = nn.ReLU(True)
        
        self.max_pool1 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.dropout4 =  nn.Dropout(0)
        self.relu4 = nn.ReLU(True)
        
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.batch_norm5 = nn.BatchNorm1d(256)
        self.dropout5 =  nn.Dropout(0)
        self.relu5 = nn.ReLU(True)
        
        self.max_pool2 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(1792, 256)
        self.relu6 = nn.ReLU(True)

        self.fc2 = nn.Linear(256, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
       
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
       

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.relu3(x)
        x = self.max_pool1(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.relu5(x)
        x = self.max_pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu6(x)

        x = self.fc2(x)

        return x

class ActRecCurve(nn.Module):
    def __init__(self, num_classes, fix_points):
        super(ActRecCurve, self).__init__()
        self.conv1 = curves.Conv1d(6, 256, kernel_size=(3,), padding=(1,), fix_points=fix_points)
        self.batch_norm1 = curves.BatchNorm1d(256, fix_points=fix_points)
        self.relu1 = nn.ReLU(True)

        self.conv2 = curves.Conv1d(256, 256, kernel_size=(3,), padding=(1,), fix_points=fix_points)
        self.batch_norm2 = curves.BatchNorm1d(256, fix_points=fix_points)
        self.dropout2 =  nn.Dropout(0)
        self.relu2 = nn.ReLU(True)
        
        self.conv3 = curves.Conv1d(256, 256, kernel_size=(3,), padding=(1,), fix_points=fix_points)
        self.batch_norm3 = curves.BatchNorm1d(256, fix_points=fix_points)
        self.dropout3 =  nn.Dropout(0)
        self.relu3 = nn.ReLU(True)
        
        self.max_pool1 = nn.MaxPool1d(2)

        self.conv4 = curves.Conv1d(256, 256, kernel_size=(3,), padding=(1,), fix_points=fix_points)
        self.batch_norm4 = curves.BatchNorm1d(256, fix_points=fix_points)
        self.dropout4 =  nn.Dropout(0)
        self.relu4 = nn.ReLU(True)
        
        self.conv5 = curves.Conv1d(256, 256, kernel_size=(3,), padding=(1,), fix_points=fix_points)
        self.batch_norm5 = curves.BatchNorm1d(256, fix_points=fix_points)
        self.dropout5 =  nn.Dropout(0)
        self.relu5 = nn.ReLU(True)
        
        self.max_pool2 = nn.MaxPool1d(2)

        self.fc1 = curves.Linear(1792, 256, fix_points=fix_points)
        self.relu6 = nn.ReLU(True)

        self.fc2 = curves.Linear(256, num_classes, fix_points=fix_points)

       # Initialize weights
        for m in self.modules():
            if isinstance(m, curves.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
                    getattr(m, 'bias_%d' % i).data.zero_()

    def forward(self, x, coeffs_t):
        x = self.conv1(x, coeffs_t)
        x = self.batch_norm1(x, coeffs_t)
        x = self.relu1(x)
       

        x = self.conv2(x, coeffs_t)
        x = self.batch_norm2(x, coeffs_t)
        x = self.dropout2(x)
        x = self.relu2(x)
        
        x = self.conv3(x, coeffs_t)
        x = self.batch_norm3(x, coeffs_t)
        x = self.dropout3(x)
        x = self.relu3(x)
        x = self.max_pool1(x)

        x = self.conv4(x, coeffs_t)
        x = self.batch_norm4(x, coeffs_t)
        x = self.dropout4(x)
        x = self.relu4(x)
        
        x = self.conv5(x, coeffs_t)
        x = self.batch_norm5(x, coeffs_t)
        x = self.dropout5(x)
        x = self.relu5(x)
        x = self.max_pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x, coeffs_t)
        x = self.relu6(x)

        x = self.fc2(x, coeffs_t)

        return x



class ActRec:
    base = ActRecBase
    curve = ActRecCurve
    kwargs = {
    }
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
#model = ActRecBase(9).to(device)
#summary(model, (6,30))
