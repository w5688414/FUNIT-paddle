import paddle.nn as nn

class ContentEncoder(nn.Layer):
    def __init__(self, input_nc=3, nf=64, n_downsampling=3, n_blocks=2):
        assert(n_blocks >= 0)
        super(ContentEncoder, self).__init__()
        self.input_nc = input_nc
        self.nf = nf
        self.n_blocks = n_blocks
        self.n_downsampling = n_downsampling

        # 7x7 conv
        layers = [nn.Pad2D(3),
                  nn.Conv2D(input_nc, nf, kernel_size=7, stride=1, padding=0),
                  nn.InstanceNorm2D(nf),
                  nn.ReLU()]

        # DownConv
        for i in range(n_downsampling):
            mult = 2 ** i
            layers += [nn.Pad2D(1),
                       nn.Conv2D(nf * mult, nf * mult * 2, kernel_size=4, stride=2, padding=0),
                       nn.InstanceNorm2D(nf * mult * 2),
                       nn.ReLU()]
        mult = 2 ** n_downsampling

        # ResBlock
        for i in range(n_blocks):
            layers += [ResnetBlock(nf * mult)]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        output = self.layers(input)

        return output


class ClassEncoder(nn.Layer):
    def __init__(self, input_nc=3, nf=64, class_dim=64, n_downsampling=4):
        super(ClassEncoder, self).__init__()
        self.input_nc = input_nc
        self.nf = nf
        self.class_dim = class_dim
        self.n_downsampling = n_downsampling

        # 7x7 conv
        layers = [nn.Pad2D(3),
                  nn.Conv2D(input_nc, nf, kernel_size=7, stride=1, padding=0),
                  nn.ReLU()]

        # DownConv
        for i in range(2):
            mult = 2 ** i
            layers += [nn.Pad2D(1),
                       nn.Conv2D(nf * mult, nf * mult * 2, kernel_size=4, stride=2, padding=0),
                       nn.ReLU()]

        mult = 2 ** 2
        for i in range(n_downsampling - 2):
            layers += [nn.Pad2D(1),
                       nn.Conv2D(nf * mult, nf * mult, kernel_size=4, stride=2, padding=0),
                       nn.ReLU()]

        layers += [nn.AdaptiveAvgPool2D(1),
                   nn.Conv2D(nf * mult, class_dim, kernel_size=1, stride=1, padding=0)]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        output = self.layers(input)

        return output.squeeze(3).squeeze(2)


class Decoder(nn.Layer):
    def __init__(self, output_nc=3, nf=512, nmf=256, class_dim=64, n_upsampling=3, n_blocks=4, mlp_blocks=3):
        assert(n_blocks >= 0)
        super(Decoder, self).__init__()
        self.output_nc = output_nc
        self.nf = nf
        self.nmf = nmf
        self.class_dim = class_dim
        self.n_upsampling = n_upsampling
        self.n_blocks = n_blocks
        self.mlp_blocks = mlp_blocks

        # MLP
        MLP = [nn.Linear(class_dim, nmf),
               nn.ReLU()]
        for i in range(1, mlp_blocks - 1):
            MLP += [nn.Linear(nmf, nmf),
                    nn.ReLU()]

        self.MLP = nn.Sequential(*MLP)

        # UpResBlock
        for i in range(n_blocks):
            setattr(self, 'Gamma' + str(i + 1) + '_1', nn.Linear(nmf, nf))
            setattr(self, 'Beta' + str(i + 1) + '_1', nn.Linear(nmf, nf))
            setattr(self, 'Gamma' + str(i + 1) + '_2', nn.Linear(nmf, nf))
            setattr(self, 'Beta' + str(i + 1) + '_2', nn.Linear(nmf, nf))
            setattr(self, 'ResBlock' + str(i + 1), ResnetAdaINBlock(nf))

        # UpConv
        layers = []
        for i in range(n_upsampling):
            mult = 2 ** i
            layers += [nn.Upsample(scale_factor=2, mode='nearest'),
                       nn.Pad2D(2),
                       nn.Conv2D(nf // mult, nf // mult // 2, kernel_size=5, stride=1, padding=0),
                       nn.InstanceNorm2D(nf // mult // 2),
                       nn.ReLU()]

        mult = 2 ** n_upsampling
        layers += [nn.Pad2D(3),
                  nn.Conv2D(nf // mult, output_nc, kernel_size=7, stride=1, padding=0),
                  nn.Tanh()]

        self.layers = nn.Sequential(*layers)

    def forward(self, input, code):
        x = input
        code = self.MLP(code)
        for i in range(self.n_blocks):
            gamma1 = getattr(self, 'Gamma' + str(i + 1) + '_1')(code)
            beta1 = getattr(self, 'Beta' + str(i + 1) + '_1')(code)
            gamma2 = getattr(self, 'Gamma' + str(i + 1) + '_2')(code)
            beta2 = getattr(self, 'Beta' + str(i + 1) + '_2')(code)
            x = getattr(self, 'ResBlock' + str(i + 1))(x, gamma1, beta1, gamma2, beta2)

        # Up-sampling
        output = self.layers(x)

        return output


class Discriminator(nn.Layer):
    def __init__(self, input_nc=3, output_nc=1, nf=64, n_blocks=10):
        assert (n_blocks >= 0)
        super(Discriminator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.nf = nf
        self.n_blocks = n_blocks

        layers = [nn.Pad2D(3),
                nn.Conv2D(input_nc, nf, kernel_size=7, stride=1, padding=0)]
        for i in range(n_blocks // 2 - 1):
            mult = 2 ** i
            layers += [PreActResnetBlock(nf * mult, nf * mult * 2),
                       PreActResnetBlock(nf * mult * 2),
                       nn.AvgPool2D(kernel_size=3, stride=2)]

        mult = 2 ** (n_blocks // 2 - 1)
        for i in range(n_blocks - (n_blocks // 2 - 1) * 2):
            layers += [PreActResnetBlock(nf * mult)]
        self.last_layer = nn.Conv2D(nf * mult, output_nc, kernel_size=1, stride=1, padding=0)
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        output = self.last_layer(nn.functional.leaky_relu(x, 0.2))

        return output

    def forward_with_features(self, input):
        features = self.layers(input)
        output = self.last_layer(nn.functional.leaky_relu(features, 0.2))

        return output, features


class ResnetBlock(nn.Layer):
    def __init__(self, in_nc, out_nc=None, use_bias=True):
        super(ResnetBlock, self).__init__()
        if out_nc is None:
            out_nc = in_nc
        conv_block = []
        conv_block += [nn.Pad2D(1),
                       nn.Conv2D(in_nc, out_nc, kernel_size=3, stride=1, padding=0, bias_attr=use_bias),
                       nn.InstanceNorm2D(out_nc),
                       nn.ReLU()]

        conv_block += [nn.Pad2D(1),
                       nn.Conv2D(out_nc, out_nc, kernel_size=3, stride=1, padding=0, bias_attr=use_bias),
                       nn.InstanceNorm2D(out_nc)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, input):
        output = input + self.conv_block(input)
        return output


class PreActResnetBlock(nn.Layer):
    def __init__(self, in_nc, out_nc=None, use_bias=True):
        super(PreActResnetBlock, self).__init__()
        self.in_nc = in_nc
        if out_nc is None:
            out_nc = in_nc
        self.out_nc = out_nc

        if not in_nc == out_nc:
            self.shortcut = nn.Conv2D(in_nc, out_nc, kernel_size=1, stride=1, padding=0, bias_attr=False)

        conv_block = []
        conv_block += [nn.LeakyReLU(0.2),
                       nn.Pad2D(1),
                       nn.Conv2D(in_nc, in_nc, kernel_size=3, stride=1, padding=0, bias_attr=use_bias)]

        conv_block += [nn.LeakyReLU(0.2),
                       nn.Pad2D(1),
                       nn.Conv2D(in_nc, out_nc, kernel_size=3, stride=1, padding=0, bias_attr=use_bias)]

        self.conv_block = nn.Sequential(*conv_block)


    def forward(self, input):
        output = self.conv_block(input)
        if not self.in_nc == self.out_nc:
            input = self.shortcut(input)

        output = input + output

        return output


class ResnetAdaINBlock(nn.Layer):
    def __init__(self, dim, use_bias=True):
        super(ResnetAdaINBlock, self).__init__()
        self.pad1 = nn.Pad2D(1)
        self.conv1 = nn.Conv2D(dim, dim, kernel_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaIN(dim)
        self.relu1 = nn.ReLU()

        self.pad2 = nn.Pad2D(1)
        self.conv2 = nn.Conv2D(dim, dim, kernel_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaIN(dim)

    def forward(self, input, gamma1, beta1, gamma2, beta2):
        x = input
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x, gamma1, beta1)
        x = self.relu1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        output = input + self.norm2(x, gamma2, beta2)

        return output


class adaIN(nn.Layer):
    def __init__(self, input_nc):
        super(adaIN, self).__init__()
        self.norm = nn.InstanceNorm2D(input_nc)

    def forward(self, input, gamma, beta):
        out_in = self.norm(input)
        out = out_in * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out
