import matplotlib.pyplot as plt
import re, sys
from pandas import DataFrame

def parse_log(file):
    iter_index_pattern = re.compile(('iter: (\d+)'))
    lr_index_pattern = re.compile(('lr: (\d+[.]\d+)'))
    pattern = re.compile('([^\s]+): (\d+[.]\d+) \((\d+.\d+)\)')

    parse_train = None
    parse_val = None

    i = 0
    with open(file, 'r') as fp:
        for line in fp:
            if "maskrcnn_benchmark.trainer INFO" in line:
                lr = re.findall(lr_index_pattern, line)
                iter = re.findall(iter_index_pattern, line)
                loss = re.findall(pattern, line)

                if len(iter)>0 and len(loss)>0:
                    train_vec = [float(iter[0]), float(lr[0]), 0.0]
                    val_vec = [float(iter[0]), float(lr[0]), 0.0]
                    header = ['iter', 'lr', 'weighted_lr']

                    for tup in loss:
                        header.append(tup[0])
                        train_vec.append(float(tup[1]))
                        val_vec.append(float(tup[2]))

                    if parse_train is None:
                        parse_train = DataFrame(columns=header)
                        parse_val = DataFrame(columns=header)

                    try:
                        parse_train.loc[i] = train_vec
                        parse_val.loc[i] = val_vec
                        i += 1
                    except ValueError:
                        print(train_vec)

    parse_train.weighted_lr = parse_train.lr * max(parse_train.loss) / max(parse_train.lr)

    ax = parse_train.plot(x='iter', y=['loss','weighted_lr'])
    parse_val.plot(ax=ax, x='iter', y=['loss'])
    plt.title(file)
    #plt.yscale("log")
    plt.show()

    parse_train.plot(x='iter', y=parse_train.columns[4:-2])
    plt.yscale("log")
    plt.title(file)
    plt.show()


    print("Done")


if __name__ == "__main__":
    #logfile = sys.argv[1]
    logfile = "output/text_experiments_refcocog.txt"
    parse_log(logfile)