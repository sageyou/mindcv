import argparse
import ast

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import nn

from mindcv.data import create_transforms
from mindcv.models import create_model

if __name__ == "__main__":
   
   ms.set_context(mode=ms.PYNATIVE_MODE)
   #=========================mae_finetune==============================================================
   
   # model_old_name = "mae_b_16_224_finetune_old"
   # model_old = create_model(
   #      model_name=model_old_name,
   #      pretrained=False,
   #      checkpoint_path = './ckpt/mae_b_16_224_finetune-cc05b899.ckpt',

   #  )
   # print("===================model_old=================")
#    print(model_old)

#    model_new_name = "mae_b_16_224_finetune_new"
#    model_new = create_model(
#         model_name=model_new_name,
#         pretrained=False,
#         checkpoint_path = './ckpt/mae_b_16_224_finetune-cc05b899.ckpt',

#     )
# #    print("===================model_new=================")
# #    print(model_new)

#=========================mae_pretrain=======================================================================

   # model_old_name = "mae_b_16_224_pretrain_old"
   
   # model_old = create_model(
   #      model_name=model_old_name,
   #      pretrained=False,
        
   #  )
   # print("===================model_old=================")
   # print(model_old)

   # model_new_name = "mae_b_16_224_pretrain_new"
   # model_vit_old_name = "vit_b_16_224"
   # model_new = create_model(
   #      model_name=model_new_name,
   #      pretrained=False,
        
   #  )
   # print("===================model_new=================")
   # print(model_new)

#==============================================vit=======================================================
# vit_l_16_224
# vit_b_32_224
# vit_l_32_224
   # ckpt_path = "./ckpt/vit_l_16_224-97d0fdbc.ckpt"
   # ckpt_path = "./vit.ckpt"
   # model_vit_old_name = "vit_l_16_224"
   # model_old = create_model(
   #      model_name=model_vit_old_name,
   #      pretrained=False,
   #      checkpoint_path = ckpt_path        
   # )
   # print("===================model_old=================")
   # print(model_old)

   print("===================model_new==============================================")
   # model_vit_new_name = "vit_l_16_224_new" 
   # ckpt_path = "./vit_ckpt_converted/vit_l_16_224.ckpt"
   # model_vit_new_name = "vit_l_32_224_new" 
   # ckpt_path = "./vit_ckpt_converted/vit_l_32_224.ckpt"
   model_vit_new_name = "vit_b_32_224_new" 
   ckpt_path = "./vit_ckpt_converted/vit_b_32_224.ckpt"

   model_new = create_model(
        model_name=model_vit_new_name,
        pretrained=False,
        checkpoint_path = ckpt_path
   )
  
