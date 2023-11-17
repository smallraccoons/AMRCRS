import pandas as pd
# 读取文本文件
data = []

if __name__ == '__main__':
    with open('record.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                epoch = line.split('|')[0].split(':')[1].strip()
                acc = line.split('|')[1].split(':')[1].strip()
                data.append({'Epoch': epoch, 'ACC': acc})

    # 创建DataFrame对象
    df = pd.DataFrame(data)

    # 将DataFrame保存为Excel文件
    df.to_excel('output.xlsx', index=False)
