import os

a = 0
b=0
while (a<2):  # 1024为我们的类别数

    dir = 'C:/Users/cyw/Desktop/test/'+str(a)+'/'  # 图片文件的地址
    label = b
    # os.listdir的结果就是一个list集，可以使用list的sort方法来排序。如果文件名中有数字，就用数字的排序
    files = os.listdir(dir)  # 列出dirname下的目录和文件
    files.sort()  # 排序
    train = open('C:/Users/cyw/Desktop/train.txt', 'a')
    text = open('C:/Users/cyw/Desktop/test.txt', 'a')
    i = 1
    for file in files:
        if i < 0:
            fileType = os.path.split(file)  # os.path.split()：按照路径将文件名和路径分割开
            if fileType[1] == '.txt':
                continue
            name = str(dir) + file + ' ' + str(int(label)) + '\n'
            train.write(name)
            i = i + 1

        else:
            fileType = os.path.split(file)
            if fileType[1] == '.txt':
                continue
            name = str(dir) + file + ' ' + str(int(label)) + '\n'
            text.write(name)
            i = i + 1
    text.close()
    train.close()
    a=a+1
    b=b+1
