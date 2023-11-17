import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import numpy as np
from torch.autograd import Variable

from utils import get_alphabet


class Bottleneck(nn.Module):

    def __init__(self, input_dim):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 1)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(input_dim, input_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    '''
    构成ResNet的残差块
    '''

    def __init__(self, inplanes, planes, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_in, block, layers, args):
        super(ResNet, self).__init__()

        self.args = args

        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer1_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer1 = self._make_layer(block, 128, 256, layers[0])
        self.layer1_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer1_bn = nn.BatchNorm2d(256)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer2 = self._make_layer(block, 256, 256, layers[1])
        self.layer2_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer2_bn = nn.BatchNorm2d(256)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer3 = self._make_layer(block, 256, 512, layers[2])
        self.layer3_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer3_bn = nn.BatchNorm2d(512)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.layer4_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer4 = self._make_layer(block, 512, 512, layers[3])
        self.layer4_conv2 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.layer4_conv2_bn = nn.BatchNorm2d(1024)
        self.layer4_conv2_relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, inplanes, planes, blocks):

        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 3, 1, 1),
                nn.BatchNorm2d(planes), )
        else:
            downsample = None
        layers = []
        layers.append(block(inplanes, planes, downsample))
        # self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes, planes, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.layer1_pool(x)
        x = self.layer1(x)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = self.layer1_relu(x)

        if not self.args.radical:
            x = self.layer2_pool(x)
        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)

        if not self.args.radical:
            x = self.layer3_pool(x)
        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)

        # x = self.layer4_pool(x)
        x = self.layer4(x)
        x = self.layer4_conv2(x)
        x = self.layer4_conv2_bn(x)
        x = self.layer4_conv2_relu(x)

        return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=7000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个形状为 (max_len, d_model) 的全零张量 pe
        pe = torch.zeros(max_len, d_model)

        # 创建一个形状为 (max_len, 1) 的张量，内容为从0到max_len-1的连续整数
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # 创建一个形状为 (d_model/2,) 的张量，内容为一系列指数值
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 计算 sin 和 cos 函数结果，并分别赋值给 pe 张量的奇数列和偶数列
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 在0维度上增加一个维度，并将其命名为 'pe'
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将输入张量 x 加上位置编码 pe，并进行 dropout 操作
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        super(MultiHeadedAttention, self).__init__()

        # 确保模型维度 d_model 可以被头数 h 整除
        assert d_model % h == 0

        # 计算每个头的维度大小
        self.d_k = d_model // h

        self.h = h

        # 使用克隆函数克隆线性层对象，总共克隆了 4 个线性层
        self.linears = clones(nn.Linear(d_model, d_model), 4)

        self.attn = None

        # 初始化 dropout 层
        self.dropout = nn.Dropout(p=dropout)

        self.compress_attention = compress_attention

        # 如果需要压缩注意力权重，则初始化一个线性层
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, align=None):
        # 如果存在掩码，则在第二个维度上扩展维度
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 通过线性层将 query、key、value 映射到多头的维度上，并进行转置
        # 注意：这里使用了 zip 函数同时遍历 self.linears 和 (query, key, value)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 调用 attention 函数进行注意力操作，得到结果 x 和注意力权重 attention_map
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, align=align)

        # 将结果 x 进行转置、重塑维度，并将最后一个维度展平
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        # 如果需要压缩注意力权重，则进行相应的操作
        if self.compress_attention:
            batch, head, s1, s2 = attention_map.shape
            attention_map = attention_map.permute(0, 2, 3, 1).contiguous()
            attention_map = self.compress_attention_linear(attention_map).permute(0, 3, 1, 2).contiguous()

        # 对最后一个线性层进行映射，并返回结果和注意力权重
        return self.linears[-1](x), attention_map


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None, align=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# 定义一个生成器类继承自nn.Module类
class Generator(nn.Module):

    # 初始化函数，d_model为特征向量维度，vocab为词表大小
    def __init__(self, d_model, vocab):
        # 调用父类初始化函数
        super(Generator, self).__init__()
        # 定义一个线性层，将特征向量映射成词表大小的向量
        self.proj = nn.Linear(d_model, vocab)
        # 定义ReLU激活函数，防止输出出现负值
        self.relu = nn.ReLU()

    # 前向传播函数，x为输入特征向量
    def forward(self, x):
        # 使用线性层将特征向量映射成词表大小的向量
        return self.proj(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 创建一个嵌入层对象lut，其中vocab表示词汇表的大小，d_model表示嵌入向量的维度（9254--》512）
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 使用lut将输入张量x转换为嵌入向量，并乘以math.sqrt(self.d_model)进行缩放。
        embed = self.lut(x) * math.sqrt(self.d_model)
        return embed


class TransformerBlock(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, ff_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadedAttention(h=num_heads, d_model=dim, dropout=dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionwiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x)
        h = self.drop(self.proj(self.attn(x, x, x, mask)[0]))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        # 初始化第一个多头注意力层，用于处理自注意力操作
        self.mask_multihead = MultiHeadedAttention(h=4, d_model=1024, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=1024)

        # 初始化第二个多头注意力层，用于处理文本和图像之间的注意力操作
        self.multihead = MultiHeadedAttention(h=4, d_model=1024, dropout=0.1, compress_attention=False)
        self.mul_layernorm2 = LayerNorm(features=1024)

        # 初始化位置前馈网络层
        self.pff = PositionwiseFeedForward(1024, 2048)
        self.mul_layernorm3 = LayerNorm(features=1024)

    def forward(self, text, conv_feature):
        # 获取文本序列的最大长度
        text_max_length = text.shape[1]

        # 生成用于掩码的遮罩矩阵
        mask = subsequent_mask(text_max_length).cuda()

        result = text

        # 使用第一个多头注意力层进行自注意力操作，并将结果与原始输入相加后进行 Layer Normalization
        result = self.mul_layernorm1(result + self.mask_multihead(result, result, result, mask=mask)[0])

        b, c, h, w = conv_feature.shape

        # 将卷积特征重塑为适合多头注意力层输入的形状
        conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()

        # 使用第二个多头注意力层将文本序列与图像特征进行注意力操作，并将结果与原始输入相加后进行 Layer Normalization
        word_image_align, attention_map = self.multihead(result, conv_feature, conv_feature, mask=None)
        result = self.mul_layernorm2(result + word_image_align)

        # 使用位置前馈网络对结果进行处理，并将结果与原始输入相加后进行 Layer Normalization
        result = self.mul_layernorm3(result + self.pff(result))

        # 返回处理后的结果以及注意力矩阵
        return result, attention_map


class Transformer(nn.Module):

    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        # 获取字符表,并映射成字典 7937
        alphabet = get_alphabet(args, 'char')
        self.word_n_class = len(alphabet)
        # 定义词嵌入层
        self.embedding_word = Embeddings(512, self.word_n_class)
        # 定义位置编码层
        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=7000)
        # 定义编码器层
        self.encoder = ResNet(num_in=3, block=BasicBlock, layers=[3, 4, 6, 3], args=args).cuda()
        # 定义解码器层
        self.decoder = Decoder()
        # 定义生成器层,将输入位数映射成词表大小
        self.generator_word = Generator(1024, self.word_n_class)

        # 如果开启了偏旁部首，则需要定义相关的层
        if args.radical:
            # 获取偏旁部首表，并映射成词典924
            radical_alphabet = get_alphabet(args, 'radical')
            self.radical_n_class = len(radical_alphabet)
            sijiaobianma_alphabet = get_alphabet(args, 'sijiaobianma')
            self.sijiaobianma_n_class = len(sijiaobianma_alphabet)
            # 定义注意力层压缩线性层
            self.attention_compress = nn.Linear(4, 1)
            # 定义特征压缩卷积层，有效地减少输入特征图的维度，从而降低后续计算的复杂性，并提取出较为重要的特征信息。（512--》64）
            self.features_compress = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0)
            # 定义偏旁嵌入层（512--》924）
            self.embedding_radical = Embeddings(512, self.radical_n_class)
            # 四角编码嵌入层
            self.embedding_sijiaobianma = Embeddings(512, self.sijiaobianma_n_class)
            # 定义偏旁部首位置编码层
            self.pe_radical = PositionalEncoding(d_model=512, dropout=0.1, max_len=7000)
            # 定义四角比那吗的位置编码层
            self.pe_sijiaobianma = PositionalEncoding(d_model=512, dropout=0.1, max_len=7000)
            # 定义偏旁部首解码器层
            self.decoder_radical = Decoder()
            # 定义四角编码解码器层
            self.decoder_sijiaobianma = Decoder()
            # 定义偏旁部首生成器层
            self.generator_radical = Generator(1024, self.radical_n_class)
            # 定义四角编码生成器层
            self.generator_sijiaobianma = Generator(1024, self.sijiaobianma_n_class)

            # 定义特征通道压缩线性层，将8*8的特征图压缩成1024个通道
            self.sub_channel = nn.Linear(2048, 1024)
            # 定义Transformer块
            self.vit_radical = TransformerBlock(1024, 4, 2048, dropout=0.1)
            # 定义分类令牌
            self.class_token = nn.Parameter(torch.zeros(1, 1, 1024))
            # 定义偏旁部首总体位置编码层
            self.pe_radical_overall = PositionalEncoding(d_model=1024, dropout=0.1, max_len=7000)
            # 定义偏旁部首总体位置编码层
            self.pe_sijiaobianma_overall = PositionalEncoding(d_model=1024, dropout=0.1, max_len=7000)
            # 定义特征通道压缩线性层2，将8*8的特征图压缩成1024个通道
            self.sub_channel_1 = nn.Linear(2048, 1024)
            # 对预测结果进行线性变换
            self.pre_logits = nn.Linear(1024, 1024)
            # 对输出进行标准化
            self.norm = nn.LayerNorm(1024, eps=1e-6)
            # 定义全连接层
            self.fc = nn.Linear(1024, self.word_n_class)
            # 定义总体生成器层
            self.generator_word_overall = Generator(1024, self.word_n_class)

    # 前向传播函数
    def forward(self, image, text_length, text_input, radical_length=None, radical_input=None, length_sijiaobianma=None,
                sijiaobianma_input=None,
                conv_feature=None, test=False, att_map=None):

        # 如果没有经过特征提取，则先将输入的图片进行特征提取
        if conv_feature is None:
            conv_feature = self.encoder(image)

        # 如果文本长度为None，则返回卷积特征图
        if text_length is None:
            return {
                'conv': conv_feature,
            }

        # 对文本进行嵌入和位置编码
        text_embedding = self.embedding_word(text_input)
        postion_embedding = self.pe(torch.zeros(text_embedding.shape).cuda()).cuda()
        text_input_with_pe = torch.cat([text_embedding, postion_embedding], 2)
        # 将文本输入解码器层和注意力层，得到输出结果和注意力矩阵
        text_input_with_pe, attention_map = self.decoder(text_input_with_pe, conv_feature)
        # 将解码器层的输出结果输入生成器层，得到预测结果
        word_decoder_result = self.generator_word(text_input_with_pe)

        # 如果不是测试，并且开启了偏旁部首
        if test == False and self.args.radical:
            # 对注意力矩阵进行处理，压缩成1个通道
            attention_map_tmp = attention_map.permute(0, 2, 3, 1)
            attention_map_tmp = self.attention_compress(attention_map_tmp).squeeze(3)  # batch * len * (H*W)
            b, c, h, w = conv_feature.size()  # H W => 8 * 64
            conv_feature_tmp = conv_feature.view(b, c, -1)

            # 将注意力矩阵和特征图进行点乘操作，得到字级别的特征图
            char_maps = torch.mul(attention_map_tmp.unsqueeze(2), conv_feature_tmp.unsqueeze(1))
            # 对得到的字级别特征图进行压缩
            char_maps = self.features_compress(char_maps.permute(0, 3, 1, 2)).permute(0, 2, 3,
                                                                                      1)  # batch * len * channel * (H*W)
            b, l, c, hw = char_maps.size()
            # 调用了.contiguous()方法，以确保张量在内存中是连续的。
            # 通过.view()方法，将char_maps张量的形状重新调整为(b * l, c * 8 * 8)，这样每个图像就被展平为一个形状为(1, c * 8 * 8)的向量
            char_maps = char_maps.contiguous().view(b * l, c, 8, 8)

            # 对偏旁部首进行嵌入和位置编码

            b, l, rl = radical_input.size()
            embedding_radical = self.embedding_radical(radical_input.view(-1, rl))
            postion_embedding_radical = self.pe_radical(torch.zeros(embedding_radical.shape).cuda()).cuda()
            radical_input_with_pe = torch.cat([embedding_radical, postion_embedding_radical], 2)
            # 将偏旁部首输入解码器层和字级别的特征图，得到输出结果和注意力矩阵
            radical_input_with_pe, attention_map = self.decoder_radical(radical_input_with_pe, char_maps)
            # 将解码器层的输出结果输入生成器层，得到预测结果 (256,12,924)
            radical_decoder_result = self.generator_radical(radical_input_with_pe)

            # 对四角编码惊醒嵌入和位置编码
            b, l, rl = sijiaobianma_input.size()
            embedding_sijiaobianma = self.embedding_sijiaobianma(sijiaobianma_input.view(-1, rl))
            postion_embedding_sijiaobianma = self.pe_sijiaobianma(
                torch.zeros(embedding_sijiaobianma.shape).cuda()).cuda()
            sijiaobianma_input_with_pe = torch.cat([embedding_sijiaobianma, postion_embedding_sijiaobianma], 2)
            # 将偏旁部首输入解码器层和字级别的特征图，得到输出结果和注意力矩阵
            sijiaobianma_input_with_pe, attention_map_sijiaobianma = self.decoder_sijiaobianma(
                sijiaobianma_input_with_pe, char_maps)
            # 将解码器层的输出结果输入生成器层，得到预测结果
            sijiaobianma_decoder_result = self.generator_sijiaobianma(sijiaobianma_input_with_pe)

        # 如果是测试，则返回预测结果、注意力矩阵和卷积特征图
        if test:
            return {
                'pred': word_decoder_result,
                'map': attention_map,
                # 'pred_sijiaobianma': sijiaobianma_decoder_result,
                'conv': conv_feature,
            }

        # 如果不是测试，则将预测结果拼接在一起返回
        else:
            total_length = torch.sum(text_length).data
            probs_res = torch.zeros(total_length, self.word_n_class).type_as(word_decoder_result.data)
            start = 0
            # （）
            for index, length in enumerate(text_length):
                length = length.data
                probs_res[start:start + length, :] = word_decoder_result[index, 0:0 + length, :]
                start = start + length

            probs_res_radical = None
            probs_res_sijiaobianma = None
            if self.args.radical:
                total_radical_length = torch.sum(radical_length.view(-1)).data
                # （1259(笔画总数)， 924（笔画的分类））
                probs_res_radical = torch.zeros(total_radical_length, self.radical_n_class).type_as(
                    radical_decoder_result.data)

                start = 0
                for index, length in enumerate(radical_length.view(-1)):
                    length = length.data
                    if length == 0:
                        continue
                    # (512，12，924)--》(1269(笔画总数),924（类别）)
                    probs_res_radical[start:start + length, :] = radical_decoder_result[index, 0:0 + length, :]
                    start = start + length

                total_sijiaobianma_length = torch.sum(length_sijiaobianma.view(-1)).data
                # （1259(四角码总数)， 10（四角码的分类））
                probs_res_sijiaobianma = torch.zeros(total_sijiaobianma_length, self.sijiaobianma_n_class).type_as(
                    sijiaobianma_decoder_result.data)

                start2 = 0
                for index, length in enumerate(length_sijiaobianma.view(-1)):
                    length2 = length.data
                    if length2 == 0:
                        continue
                    # (512，12，10)--》(1269(笔画总数),10（类别）)
                    probs_res_sijiaobianma[start2:start2 + length2, :] = sijiaobianma_decoder_result[index,
                                                                         0:0 + length2, :]
                    start2 = start2 + length2

            return {
                'pred': probs_res,
                'radical_pred': probs_res_radical,
                'sijiaobianma_pred': probs_res_sijiaobianma,
                'map': attention_map,
                'conv': conv_feature,
            }


if __name__ == '__main__':
    net = ResNet(num_in=3, block=BasicBlock, layers=[3, 4, 6, 3]).cuda()
    image = torch.Tensor(8, 3, 64, 64).cuda()
    result = net(image)
    print(result.shape)
    pass
