from . import unet

albu_models = {
    'albu_resnet34_upsample': unet.Resnet34_upsample,
    'albu_resnet34_sum': unet.Resnet34_sum,
    'albu_resnet34_double': unet.Resnet34_double,
    'albu_resnet34_bn': unet.Resnet34_bn_sum,
    'albu_resnet34_dil': unet.DilatedResnet34,
    'albu_resnet50bn': unet.Resnet50bn_upsample,
    'albu_dpn': unet.DPNUnet,
    'albu_incv3': unet.Incv3,
    'albu_inc_resnet': unet.LinkNetIncvRes2,
    'albu_vgg11bn': unet.Vgg11bn,
    'albu_vgg16bn': unet.Vgg16bn,
    'unet_se_resnext50': unet.SE_ResNeXt_50,
    'unet_densenet161': unet.DenseNet161,
    'unet_densenet121': unet.DenseNet121,
    'unet_densenet169': unet.DenseNet169,
    'unet_densenet201': unet.DenseNet201,
    'unet_resnext50': unet.ResNeXt50
}
