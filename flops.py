from ptflops import get_model_complexity_info
from dataset import *
from models.model import *
from utils import *
from models.modified1 import *
from models.modified2 import *
from models.modified3 import *
from models.model_sharedAttention import SharedSwinTransformerSys_modified
from models.CA import *
from models.model_sharedAttention_MLP import SharedSwinTransformerSys_MLP_modified



model = SwinUnet_modified_shared_MLP(img_size=IMG_SIZE,num_classes=num_classes)
flops, params = get_model_complexity_info(model, (1, 512, 512), as_strings=True,
                                          print_per_layer_stat=False) 
print("Trans3NuSeg with attention sharing with token MLP, flop {}, params {}".format(flops,params))
