import torch
from data.dataset import lmdbDataset, resizeNormalize
import pickle as pkl

global alp2num_character, alphabet_character, alp2num_radical, alphabet_radical, char2radicals, alp2num_sijiaobianma, alphabet_sijiaobianma, char2sijiaobianma
alp2num_character = None


def get_dataloader(root, args, shuffle=False):
    if root.endswith('pkl'):
        f = open(root, 'rb')
        dataset = pkl.load(f)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=8,
        )
    else:
        dataset = lmdbDataset(root, resizeNormalize((args.imageW, args.imageH)))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=8,
        )
    return dataloader, dataset


def get_data_package(args):
    test_dataset = []
    for dataset_root in args.test_dataset.split(','):
        _, dataset = get_dataloader(dataset_root, args, shuffle=True)
        test_dataset.append(dataset)
    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset_total, batch_size=args.batch_size, shuffle=False, num_workers=8,
    )

    if not args.test:
        train_dataset = []
        for dataset_root in args.train_dataset.split(','):
            _, dataset = get_dataloader(dataset_root, args, shuffle=True)
            train_dataset.append(dataset)
        train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_total, batch_size=args.batch_size, shuffle=True, num_workers=8,
        )
        return train_dataloader, test_dataloader
    else:
        return None, test_dataloader


def converter(label, args):
    string_label = label
    label = [i for i in label]
    alp2num = alp2num_character

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()
    max_length = max(length)

    text_input = torch.zeros(batch, max_length).long().cuda()
    for i in range(batch):
        for j in range(len(label[i]) - 1):
            text_input[i][j + 1] = alp2num[label[i][j]]

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            if j == (len(label[i]) - 1):
                text_all[start + j] = alp2num['END']
            else:
                text_all[start + j] = alp2num[label[i][j]]
        start += len(label[i])

    if args.radical:
        length_radical = []
        for i in range(batch):
            length_tmp = []
            for j in range(max_length):
                if j < len(label[i]) - 1:
                    length_tmp.append(len(char2radicals[label[i][j]]) + 1)
                else:
                    length_tmp.append(0)
            length_radical.append(length_tmp)
        length_radical = torch.Tensor(length_radical).long().cuda()

        length_sijiaobianma = []
        for i in range(batch):
            length_tmp2 = []
            for j in range(max_length):
                if j < len(label[i]) - 1:
                    length_tmp2.append(len(char2sijiaobianma[label[i][j]]) + 1)
                else:
                    length_tmp2.append(0)
            length_sijiaobianma.append(length_tmp2)
        length_sijiaobianma = torch.Tensor(length_sijiaobianma).long().cuda()

        max_radical_len = max(length_radical.view(-1))
        radical_input = torch.zeros(batch, max_length, max_radical_len).long().cuda()
        for i in range(batch):
            for j in range(len(label[i]) - 1):
                radicals = char2radicals[label[i][j]]
                for k in range(len(radicals)):
                    radical_input[i][j][k + 1] = k + 1

        sum_length = sum(length_radical.view(-1))
        radical_all = torch.zeros(sum_length).long().cuda()
        start = 0
        for i in range(batch):
            for j in range(len(label[i]) - 1):
                radicals = char2radicals[label[i][j]]
                for k in range(len(radicals)):
                    radical_all[start + k] = alp2num_radical[radicals[k]]
                radical_all[start + len(radicals)] = alp2num_radical['END']
                start += (len(radicals) + 1)

        max_sijiaobianma_len = max(length_sijiaobianma.view(-1))
        sijiaobianma_input = torch.zeros(batch, max_length, max_sijiaobianma_len).long().cuda()
        for i in range(batch):
            for j in range(len(label[i]) - 1):
                sijiaobianma = char2sijiaobianma[label[i][j]]
                for k in range(len(sijiaobianma)):
                    sijiaobianma_input[i][j][k + 1] = k + 1

        sum_length2 = sum(length_sijiaobianma.view(-1))
        sijiaobianma_all = torch.zeros(sum_length2).long().cuda()
        start = 0
        for i in range(batch):
            for j in range(len(label[i]) - 1):
                sijiaobianmas = char2sijiaobianma[label[i][j]]
                if len(sijiaobianmas) == 1:
                    sijiaobianma_all[start + 0] = alp2num_sijiaobianma[sijiaobianmas[0]]
                else:
                    for k in range(len(sijiaobianmas)):
                        sijiaobianma_all[start + k] = alp2num_sijiaobianma[sijiaobianmas[k]]
                sijiaobianma_all[start + len(sijiaobianmas)] = alp2num_sijiaobianma['END']
                start += (len(sijiaobianmas) + 1)


    if args.radical:
        return length, text_input, text_all, length_radical, radical_input, radical_all, string_label, length_sijiaobianma, sijiaobianma_input, sijiaobianma_all
    else:
        return length, text_input, text_all, None, None, None, string_label


def get_alphabet(args, type):
    global alp2num_character, alphabet_character, alp2num_radical, alphabet_radical, char2radicals, alp2num_sijiaobianma, alphabet_sijiaobianma, char2sijiaobianma
    if alp2num_character == None:
        alphabet_character_file = open(args.alpha_path)
        alphabet_character = list(alphabet_character_file.read().strip())
        alphabet_character_raw = ['START', '\xad']
        for item in alphabet_character:
            alphabet_character_raw.append(item)
        alphabet_character_raw.append('END')
        alphabet_character = alphabet_character_raw
        alp2num = {}
        for index, char in enumerate(alphabet_character):
            alp2num[char] = index
        alp2num_character = alp2num

        if args.radical:
            alphabet_radical_file = open(args.alpha_path_radical)
            alphabet_radical = alphabet_radical_file.readlines()
            alphabet_radical = [item.strip('\n') for item in alphabet_radical]
            alphabet_radical_raw = ['START']
            for item in alphabet_radical:
                alphabet_radical_raw.append(item)
            alphabet_radical_raw.append('END')
            alphabet_radical = alphabet_radical_raw
            alp2num_r = {}
            for index, char in enumerate(alphabet_radical):
                alp2num_r[char] = index
            alp2num_radical = alp2num_r  # 偏旁部首转换成数组，加上对应的编号

            alphabet_sijiaobianma_file = open(args.alpha_path_sijiaobianma)
            alphabet_sijiaobianma = alphabet_sijiaobianma_file.readlines()
            alphabet_sijiaobianma = [item.strip('\n') for item in alphabet_sijiaobianma]
            alphabet_sijiaobianma_raw = ['START']
            for item in alphabet_sijiaobianma:
                alphabet_sijiaobianma_raw.append(item)
            alphabet_sijiaobianma_raw.append('END')
            alphabet_sijiaobianma = alphabet_sijiaobianma_raw
            alp2num_r2 = {}
            for index, char in enumerate(alphabet_sijiaobianma):
                alp2num_r2[char] = index
            alp2num_sijiaobianma = alp2num_r2  # 四角编码得单个字符转换成数组，加上对应的编号

            files = open(args.decompose_path).readlines()
            c2r = {}
            for line in files:
                items = line.strip('\n').strip().split(':')
                c2r[items[0]] = items[1].split(' ')
            # 将c2r中不包含alphabet_character表中的文字架构，加入到数组中
            for i in range(1, len(alphabet_character) - 1):
                if alphabet_character[i] not in c2r.keys():
                    c2r[alphabet_character[i]] = alphabet_character[i]
            char2radicals = c2r

            # 对四角编码进行数据加载
            alphabet_sijiaobianma_file = open(args.sijiaobianma_path).readlines()
            c2r2 = {}
            for line in alphabet_sijiaobianma_file:
                items = line.strip('\n').strip().split(':')
                c2r2[items[0]] = items[1].split(',')
            # 将c2r2中不包含alphabet_character表中的文字架构，加入到数组中
            for i in range(1, len(alphabet_character) - 1):
                if alphabet_character[i] not in c2r2.keys():
                    # c2r2[alphabet_character[i]] = 'a'
                    c2r2[alphabet_character[i]] = alphabet_character[i]
            char2sijiaobianma = c2r2

    if type == 'char':
        return alphabet_character
    elif type == 'radical':
        return alphabet_radical
    else:
        return alphabet_sijiaobianma


def tensor2str(tensor, args):
    alphabet = get_alphabet(args, 'char')
    string = ""
    for i in tensor:
        if i == (len(alphabet) - 1):
            continue
        string += alphabet[i]
    return string
