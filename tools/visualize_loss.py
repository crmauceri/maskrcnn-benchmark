import matplotlib.pyplot as plt
import re, sys
from pandas import DataFrame

def parse_log(file):
    iter_index_pattern = re.compile(('iter: (\d+)'))
    pattern = re.compile('([^\s]+): (\d+[.]\d+) \((\d+.\d+)\)')

    parse_train = None
    parse_val = None

    i = 0
    with open(file, 'r') as fp:
        for line in fp:
            if "maskrcnn_benchmark.trainer INFO" in line:
                iter = re.findall(iter_index_pattern, line)
                loss = re.findall(pattern, line)

                if len(iter)>0 and len(loss)>0:
                    train_vec = [float(iter[0])]
                    val_vec = [float(iter[0])]
                    header = ['iter']

                    for tup in loss:
                        header.append(tup[0])
                        train_vec.append(float(tup[1]))
                        val_vec.append(float(tup[2]))

                    if parse_train is None:
                        parse_train = DataFrame(columns=header)
                        parse_val = DataFrame(columns=header)

                    parse_train.loc[i] = train_vec
                    parse_val.loc[i] = val_vec
                    i += 1

    plt.plot(parse_train.iter, parse_train.loss)
    plt.title("Total Loss")
    plt.show()

    parse_train.plot(x='iter', y=parse_train.columns[2:-2])
    plt.show()


    print("Done")


if __name__ == "__main__":
    logfile = sys.argv[1]
    parse_log(logfile)