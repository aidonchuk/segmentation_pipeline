import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet, inception

from models import inception_resnet_v2, vgg, se_resnext, densenet, resnext
from models.blocks import ConcurrentSEModule
from models.dpn import dpn92
from models_common import ConvRelu

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Upscale:
    transposed_conv = 0
    upsample_bilinear = 1
    pixel_shuffle = 2


torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
wideresnet_path = os.path.join(model_dir, 'wide_resnet38_ipabn_lr_256.pth')

encoder_params = {
    'resnet34':
        {'filters': [64, 64, 128, 256, 512],
         'init_op': resnet.resnet34,
         'url': resnet.model_urls['resnet34']},
    'resnet18':
        {'filters': [64, 64, 128, 256, 512],
         'init_op': resnet.resnet18,
         'url': resnet.model_urls['resnet18']},
    'resnet50':
        {'filters': [64, 256, 512, 1024, 2048],
         'init_op': resnet.resnet50,
         'url': resnet.model_urls['resnet50']},
    'inceptionv3':
        {'filters': [64, 192, 288, 768, 2048],
         'init_op': inception.inception_v3,
         'url': inception.model_urls['inception_v3_google']},
    'inception_resnet_v2':
        {'filters': [64, 192, 320, 1088, 1536],
         'init_op': inception_resnet_v2.inceptionresnetv2,
         'url': inception_resnet_v2.model_urls['inceptionresnetv2']},
    'vgg11_bn':
        {'filters': [64, 128, 256, 512, 512],
         'init_op': vgg.vgg11_bn,
         'url': vgg.model_urls['vgg11_bn']},
    'vgg16_bn':
        {'filters': [64, 128, 256, 512, 512],
         'init_op': vgg.vgg16_bn,
         'url': vgg.model_urls['vgg16_bn']},
    'vgg11':
        {'filters': [64, 128, 256, 512, 512],
         'init_op': vgg.vgg11,
         'url': vgg.model_urls['vgg11']},
    'classic_unet':
        {'filters': [32, 64, 128, 256, 512],
         'init_op': vgg.vgg_unet_avgpool,
         'url': None},
    'dpn92':
        {'filters': [64, 336, 704, 1552, 2688],
         'init_op': dpn92,
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth'},
    'se_resnext50':
        {'filters': [64, 256, 512, 1024, 2048],
         'init_op': se_resnext.se_resnext50,
         'url': None},
    'densenet121':
        {'filters': [64, 256, 512, 1024, 1024],
         'init_op': densenet.densenet121,
         'url': None},
    'densenet169':
        {'filters': [64, 256, 512, 1280, 1664],
         'init_op': densenet.densenet169,
         'url': None},
    'densenet201':
        {'filters': [64, 256, 512, 1792, 1920],
         'init_op': densenet.densenet201,
         'url': None},
    'densenet161':
        {'filters': [96, 384, 768, 2112, 2208],
         'init_op': densenet.densenet161,
         'url': None},
    'resnext50':
        {'filters': [64, 256, 512, 1024, 2048],
         'init_op': resnext.resnext50,
         'url': None}
}


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # InPlaceABN(out_channels)  FIXME
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class ConvBottleneck1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # InPlaceABN(out_channels)  FIXME
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class ConvBottleneckSimple(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return x


class SumBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        assert in_channels // 2 == out_channels
        super().__init__()

    def forward(self, dec, enc):
        return dec + enc


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class UnetDoubleDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class UnetBNDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # InPlaceABN(out_channels)  FIXME
        )

    def forward(self, x):
        return self.layer(x)


class UnetDecoderBlock1x1(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, padding=0),
            nn.ReLU(inplace=True)
            # InPlaceABN(out_channels)  FIXME
        )

    def forward(self, x):
        return self.layer(x)


class UnetDoubleBNDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class UnetDecoderBlockConcurrent(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.se = ConcurrentSEModule(out_channels)

    def forward(self, x):
        return self.se(self.layer(x))


class UnetDecoderBlockConcurrentBN(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = ConcurrentSEModule(out_channels)

    def forward(self, x):
        return self.se(self.layer(x))


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = torch.nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()['conv1.weight'][:, :3, ...] = pretrained_dict['conv1.weight'].data
            # model.state_dict()['Conv2d_1a_3x3.conv.weight'][:,:3,...] = pretrained_dict['Conv2d_1a_3x3.conv.weight']
            # pretrained_dict['conv1.']
            skip_layers = self.first_layer_params_names
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               not any(k.startswith(s) for s in skip_layers)}
            # todo recalc
        # print(pretrained_dict.keys())
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_names(self):
        return ['conv1.conv']


def _get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])


class EncoderDecoder(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34', dropout=None, requires_grad=True,
                 hyper_column=False):
        if not hasattr(self, 'decoder_type'):
            self.decoder_type = Upscale.upsample_bilinear
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'need_center'):
            self.need_center = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        self.filters = encoder_params[encoder_name]['filters']

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        if self.need_center:
            self.center = self.get_center()

        self.bottlenecks = nn.ModuleList(
            [self.bottleneck_type(f * 2, f) for f in reversed(self.filters[:-1])])  # todo init from type
        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(1, len(self.filters))])

        if self.first_layer_stride_two:
            middle_filters = self.filters[0]
            self.last_upsample = self.decoder_block(self.filters[0], middle_filters, self.filters[0],
                                                    upscale=self.decoder_type)

        self.dropout = None
        if dropout is not None:
            assert isinstance(dropout, float)

        self.hc = hyper_column

        self.final = self.make_final_classifier(self.filters[0], num_classes)

        # self._initialize_weights()

        encoder = encoder_params[encoder_name]['init_op'](in_channels=num_channels, requires_grad=requires_grad)
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
        if encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'], num_channels != 3)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        prt = False
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if prt: print('x_' + str(i) + ' ' + str(x.shape))
            enc_results.append(x.clone())

        if self.need_center:
            x = self.center(x)

        dec = []

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])
            if prt: print('x_' + str(idx) + ' ' + str(x.shape))
            dec.append(x)

        if self.first_layer_stride_two:
            x = self.last_upsample(x)
            if prt: print('x_' + str(len(dec) + 1) + ' ' + str(x.shape))
            dec.append(x)

        if self.hc:
            x = torch.cat([dec[4],
                           nn.Upsample(scale_factor=2, mode='bilinear')(dec[3]),
                           nn.Upsample(scale_factor=4, mode='bilinear')(dec[2]),
                           nn.Upsample(scale_factor=8, mode='bilinear')(dec[1]),
                           nn.Upsample(scale_factor=16, mode='bilinear')(dec[0])
                           ], 1)
            x = ConvRelu(1920, self.filters[0])(x)

        if self.dropout is not None:
            x = self.dropout(x)

        f = self.final(x)
        if prt: print('x_f' + ' ' + str(f.shape))
        # f = f.squeeze(1)  # FIXME ugly
        return f

    def get_decoder(self, layer):
        return self.decoder_block(self.filters[layer], self.filters[layer], self.filters[max(layer - 1, 0)],
                                  self.decoder_type)

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(
            # nn.Conv2d(in_filters // 2, in_filters // 2, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_filters, num_classes, 1, padding=0)  # kernel_size=1 3, padding=1
        )

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _get_layers_params([self.encoder_stages[0]])

    @property
    def first_layer_params_names(self):
        raise NotImplementedError

    @property
    def layers_except_first_params(self):
        layers = get_slice(self.encoder_stages, 1, -1) + [self.bottlenecks, self.decoder_stages, self.final]
        if self.need_center:
            layers += [self.center_pool, self.center]
        return _get_layers_params(layers)


def get_slice(features, start, end):
    if end == -1:
        end = len(features)
    return [features[i] for i in range(start, end)]


class Aggregator(nn.Module):
    def __init__(self, in_channels, mid_channels, upsample_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2 ** upsample_factor)
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = F.relu(self.conv(x))
        return x


class PathAggregationEncoderDecoder(EncoderDecoder):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34'):
        self.bottleneck_type = SumBottleneck
        super().__init__(num_classes, num_channels, encoder_name)

        self.aggretagors = nn.ModuleList([Aggregator(f, self.filters[0], len(self.filters) - 2 - i) for i, f in
                                          enumerate(reversed(self.filters[:-1]))])  # todo init from type
        self.aggregate = nn.Conv2d(self.filters[0], self.filters[0], 3, padding=1)

    def forward(self, x):
        # Encoder
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(x.clone())

        if self.need_center:
            x = self.center(x)

        bottleneck_results = []
        y = None
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])
            if idx < len(self.filters) - 2:
                y = self.aggretagors[idx](x)
            else:
                y = x
            bottleneck_results.append(y)

        x = self.aggregate(sum(bottleneck_results))
        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        f = self.final(x)

        return f


class DPEncoderDecoder(AbstractModel):
    # should be successor of encoder decoder
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34'):
        if not hasattr(self, 'decoder_type'):
            self.decoder_type = Upscale.upsample_bilinear
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'need_center'):
            self.need_center = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        self.filters = encoder_params[encoder_name]['filters']

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        if self.need_center:
            self.center = self.get_center()

        self.bottlenecks = nn.ModuleList(
            [self.bottleneck_type(f * 2, f) for f in reversed(self.filters[:-1])])  # todo init from type
        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(1, len(self.filters))])

        if self.first_layer_stride_two:
            middle_filters = self.filters[0]
            self.last_upsample = self.decoder_block(self.filters[0], middle_filters, self.filters[0],
                                                    upscale=self.decoder_type)

        # self.dropout = nn.Dropout2d(p=0.5)

        self.final = self.make_final_classifier(self.filters[0], num_classes)

        self._initialize_weights()

        encoder = encoder_params[encoder_name]['init_op'](in_channels=num_channels)
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
        if encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'], num_channels != 3)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        if self.need_center:
            x = self.center(x)

        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        if self.first_layer_stride_two:
            x = self.last_upsample(x)
        # x = self.dropout(x)
        f = self.final(x)
        f = f.squeeze(1)  # FIXME ugly
        return f

    def get_decoder(self, layer):
        return self.decoder_block(self.filters[layer], self.filters[layer], self.filters[max(layer - 1, 0)],
                                  self.decoder_type)

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(
            # nn.Conv2d(in_filters // 2, in_filters // 2, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_filters, num_classes, 3, padding=1)
        )
