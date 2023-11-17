import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from transocr import Transformer
from utils import get_data_package, converter, tensor2str, get_alphabet
import zhconv

parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_name', type=str, default='TransOCR', help='')
parser.add_argument('--dataset_name', type=str, default='document', help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--lr', type=float, default=1.0, help='')
parser.add_argument('--epoch', type=int, default=1000, help='')
parser.add_argument('--radical', action='store_true', default=True)
parser.add_argument('--sijiaohaoma', action='store_true', default=True)
parser.add_argument('--test', action='store_true', default=True)
# parser.add_argument('--resume', type=str, default='/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/history/TransOCR_02_90/best_model.pth', help='')
parser.add_argument('--resume', type=str, default='', help='')
parser.add_argument('--train_dataset', type=str,
                    default='/home/lab421/chencheng/benchmarking-chinese-text-recognition/data/dataset/benchmark_dataset/document/document_train',
                    help='')
parser.add_argument('--test_dataset', type=str,
                    default='/home/lab421/chencheng/benchmarking-chinese-text-recognition/data/dataset/benchmark_dataset/document/document_test',
                    help='')
parser.add_argument('--imageH', type=int, default=32, help='')
parser.add_argument('--imageW', type=int, default=256, help='')
parser.add_argument('--coeff', type=float, default=1.0, help='')
parser.add_argument('--coeff2', type=float, default=1.0, help='')
parser.add_argument('--alpha_path', type=str,
                    default='/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/data/benchmark.txt',
                    help='')
parser.add_argument('--alpha_path_radical', type=str,
                    default='/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/data/radicals.txt',
                    help='')
parser.add_argument('--alpha_path_sijiaobianma', type=str,
                    default='/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/data2/encoder2.txt',
                    help='')
parser.add_argument('--decompose_path', type=str,
                    default='/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/data/decompose.txt',
                    help='')
parser.add_argument('--sijiaobianma_path', type=str,
                    default='/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/data2/sijiaobianma_2w2.txt',
                    help='')
args = parser.parse_args()

# benchmark编程字典
alphabet = get_alphabet(args, 'char')
print('alphabet:', alphabet)

model = Transformer(args).cuda()
model = nn.DataParallel(model)
train_loader, test_loader = get_data_package(args)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss().cuda()
best_acc = -1

# 添加的参数
start_epoch = -1
RESUME = False


if args.resume.strip() != '':
    model.load_state_dict(torch.load(args.resume))
    print('loading pretrained model！！！')


def train(epoch, iteration, image, length, text_input, text_gt, length_radical, radical_input, radical_gt,
          length_sijiaobianma, sijiaobianma_input, sijiaobianma_all):
    model.train()
    optimizer.zero_grad()
    # length: 每个句子的长度，length：(32, 16),length_radical:(32,16) radical_input:笔画
    result = model(image, length, text_input, length_radical, radical_input, length_sijiaobianma, sijiaobianma_input)

    text_pred = result['pred']
    loss_char = criterion(text_pred, text_gt)  #

    if args.radical:
        radical_pred = result['radical_pred']
        loss_radical = criterion(radical_pred, radical_gt)  #
        loss = loss_char + args.coeff * loss_radical
    if args.sijiaobianma & args.radical:
        sijiaobianma_pred = result['sijiaobianma_pred']
        loss_sijiaobianma = criterion(sijiaobianma_pred, sijiaobianma_all)  #

        loss = loss_char + args.coeff * loss_radical + args.coeff2 * loss_sijiaobianma
        print(
            'epoch : {} | iter : {}/{} | loss : {} | char : {} | radical : {} | code : {} '.format(epoch, iteration,
                                                                                                   len(train_loader),
                                                                                                   loss, loss_char,
                                                                                                   loss_radical,
                                                                                                   loss_sijiaobianma))
        # loss = loss_char + args.coeff * loss_radical
        # print(
        #     'epoch : {} | iter : {}/{} | loss : {} | char : {} | radical : {}'.format(epoch, iteration,
        #                                                                                            len(train_loader), loss, loss_char, loss_radical))
    else:
        loss = loss_char
        print('epoch : {} | iter : {}/{} | loss : {}'.format(epoch, iteration, len(train_loader), loss))
    loss.backward()
    optimizer.step()


test_time = 0


@torch.no_grad()
def test(epoch):
    torch.cuda.empty_cache()
    global test_time
    test_time += 1
    torch.save(model.state_dict(),
               '/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/history/{}/{}/model.pth'.format(
                   args.dataset_name, args.exp_name))
    if epoch == -1:
        model.load_state_dict(torch.load('/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/history/{}/{}/best_model.pth'.format(
                args.dataset_name, args.exp_name))) # 加载已保存的模型参数

    result_file = open(
        '/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/history/{}/{}/result_file_test_{}.txt'.format(
            args.dataset_name, args.exp_name, test_time), 'w+', encoding='utf-8')

    print("Start Eval!")
    model.eval()
    dataloader = iter(test_loader)
    test_loader_len = len(test_loader)

    correct = 0
    total = 0

    for iteration in range(test_loader_len):
        data = next(dataloader)
        image, label, _ = data
        image = torch.nn.functional.interpolate(image, size=(args.imageH, args.imageW))
        length, text_input, text_gt, length_radical, radical_input, radical_gt, string_label, length_sijiaobianma, sijiaobianma_input, sijiaobianma_all = converter(
            label, args)
        max_length = max(length)

        batch = image.shape[0]
        pred = torch.zeros(batch, 1).long().cuda()
        image_features = None
        prob = torch.zeros(batch, max_length).float()
        for i in range(max_length):
            length_tmp = torch.zeros(batch).long().cuda() + i + 1
            result = model(image, length_tmp, pred, conv_feature=image_features, test=True)

            prediction = result['pred']
            now_pred = torch.max(torch.softmax(prediction, 2), 2)[1]
            prob[:, i] = torch.max(torch.softmax(prediction, 2), 2)[0][:, -1]
            pred = torch.cat((pred, now_pred[:, -1].view(-1, 1)), 1)
            image_features = result['conv']
        # 下面代码的作用是将text_gt划分成长度为length的小段存储到text_gt_list中
        text_gt_list = []
        start = 0
        for i in length:
            text_gt_list.append(text_gt[start: start + i])
            start += i
        # 这段代码的作用是将预测结果整理成列表，并存储在text_pred_list中；同时将每个预测结果的概率乘起来，存储在text_prob_list中
        text_pred_list = []
        # 用于存放bm
        text_prob_list = []
        for i in range(batch):
            now_pred = []
            for j in range(max_length):
                if pred[i][j] != len(alphabet) - 1:
                    now_pred.append(pred[i][j])
                else:
                    break
            text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

            overall_prob = 1.0
            for j in range(len(now_pred)):
                overall_prob *= prob[i][j]
            text_prob_list.append(overall_prob)
        # 下面代码的作用是将text_pred_list和text_gt_list两个列表中的结果逐一比较，并打印输出结果以及预测的准确率
        start = 0
        for i in range(batch):
            state = False
            pred = zhconv.convert(tensor2str(text_pred_list[i], args),
                                  'zh-cn')  # zhconv.convert()将字符串从一种中文字符集转换为另一种中文字符集
            gt = zhconv.convert(tensor2str(text_gt_list[i], args), 'zh-cn')

            if pred == gt:
                correct += 1
                state = True
            start += i
            total += 1
            print('{} | {} | {} | {} | {} | {}'.format(total, pred, gt, state, text_prob_list[i],
                                                       correct / total))
            result_file.write(
                '{} | {} | {} | {} | {} \n'.format(total, pred, gt, state, text_prob_list[i]))

    print("ACC : {}".format(correct / total))
    global best_acc
    if correct / total > best_acc:
        best_acc = correct / total
        torch.save(model.state_dict(),
                   '/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/history/{}/{}/best_model.pth'.format(
                       args.dataset_name, args.exp_name))

    f = open(
        '/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/history/{}/{}/record.txt'.format(
            args.dataset_name, args.exp_name), 'a+', encoding='utf-8')
    f.write("Epoch : {} | ACC : {}\n".format(epoch, correct / total))
    f.close()


if __name__ == '__main__':
    print('-------------')

    if not os.path.isdir(
            '/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/history/{}/{}'.format(
                args.dataset_name, args.exp_name)):
        os.mkdir('/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/history/{}/{}'.format(
            args.dataset_name, args.exp_name))
    if args.test:
        test(-1)
        exit(0)

    if RESUME:
        path_checkpoint = "./checkpoint_web/ckpt_best_scene_68.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        best_acc = checkpoint['best_acc']
        test_time = checkpoint['epoch'] + 1

    for epoch in range(start_epoch + 1, args.epoch):
        torch.save(model.state_dict(),
                   '/home/lab421/chencheng/benchmarking-chinese-text-recognition/models/TransOCR/history/{}/{}/model.pth'.format(
                       args.dataset_name, args.exp_name))
        dataloader = iter(train_loader)
        train_loader_len = len(train_loader)

        print('length of training datasets:', train_loader_len)
        for iteration in range(train_loader_len):
            data = next(dataloader)
            image, label, _ = data
            # 函数会对输入的图像进行双线性插值，从而得到与指定大小相等的新图像。
            # 在进行插值之前，该函数还会根据给定的大小在输入图像的长边和短边之间选择一个合适的比例系数，以保持图像的宽高比。
            # 插值后得到的新图像可以在后续的神经网络模型中被用作输入。
            image = torch.nn.functional.interpolate(image, size=(args.imageH, args.imageW))

            length, text_input, text_gt, length_radical, radical_input, radical_gt, string_label, length_sijiaobianma, sijiaobianma_input, sijiaobianma_all = converter(
                label,
                args)
            train(epoch, iteration, image, length, text_input, text_gt, length_radical, radical_input, radical_gt,
                  length_sijiaobianma, sijiaobianma_input, sijiaobianma_all)

        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc
        }
        if not os.path.isdir("./checkpoint"):
            os.mkdir("./checkpoint")
        torch.save(checkpoint, './checkpoint/ckpt_best_document_%s.pth' % (str(epoch)))

        test(epoch)

        # scheduler
        if (epoch + 1) <= 20 and (epoch + 1) % 8 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.8
        elif (epoch + 1) > 20 and (epoch + 1) % 2 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.8
