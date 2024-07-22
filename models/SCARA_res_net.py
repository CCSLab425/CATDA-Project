import torch
import torch.nn as nn
from models.vit import ViT
from models.SCARA_res_blo import InvertedResidual_Block

class SCARA_ViT_model(nn.Module):
    def __init__(self, input_size=32, class_num=None):
        super(SCARA_ViT_model, self).__init__()
        self.sharedNetwork = ViT(image_size=input_size, patch_size=16, num_classes=class_num, dim=1, depth=2, heads=2,
                                 mlp_dim=1, dropout=0.1, emb_dropout=0.1)

    def forward(self, x , x_y=None, y=None):  
        x, x_y, y = self.sharedNetwork(x,x_y,y)
        return x, x_y, y

if __name__ == "__main__":
    DEVICE = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
    model = SCARA_ViT_model(input_size=32, class_num=4).to(DEVICE)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    #flops, params = profile(model, (input1,))
    #summary(model, [(1, 32, 32), (1, 32, 32)])