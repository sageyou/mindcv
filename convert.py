import mindspore as ms


if __name__ == "__main__":
    def convert(checkpoint_param):
        ms_checkpoint = []
        for n in checkpoint_param:
            p = checkpoint_param[n]
            d = {}
            if n=='head.classifier.weight':
                d['name'] = 'head.weight'
            elif n=='head.classifier.bias':
                d['name'] = 'head.bias' 
            else:
                d['name'] = n
            d['data'] = p
            ms_checkpoint.append(d)
        ms.save_checkpoint(ms_checkpoint, 'vit_l_16_224.ckpt')

    ckpt_path = "./ckpt/vit_l_16_224-97d0fdbc.ckpt"
#     ckpt_path = "./ckpt/vit_l_32_224-b80441df.ckpt"
#     ckpt_path = "./ckpt/vit_b_32_224-f50866e8.ckpt"
    checkpoint_param = ms.load_checkpoint(ckpt_path)
    convert(checkpoint_param)