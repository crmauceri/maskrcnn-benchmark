import matplotlib.pyplot as plt
import re, sys
from pandas import DataFrame

def parse_log(file):
    iter_index_pattern = re.compile(('iter: (\d+)'))
    pattern = re.compile('(\d+[.]\d+)')

    parse_train = DataFrame(columns=['iter', 'loss', 'loss_objectness', 'loss_rpn_box_reg', 'text_loss',
                                     'ref_objectness', 'ref_rpn_box_reg'])
    parse_val = DataFrame(columns=['iter', 'loss', 'loss_objectness', 'loss_rpn_box_reg', 'text_loss',
                                     'ref_objectness', 'ref_rpn_box_reg'])

    i = 0
    with open(file, 'r') as fp:
        for line in fp:
            if "maskrcnn_benchmark.trainer INFO" in line:
                iter = re.findall(iter_index_pattern, line)
                nums = re.findall(pattern, line)

                if len(iter)>0 and len(nums)==17:
                    parse_train.loc[i] = list(map(float, [iter[0]] + nums[0:-5:2]))
                    parse_val.loc[i] = list(map(float, [iter[0]] + nums[1:-5:2]))
                    i += 1

    plt.plot(parse_train.iter, parse_train.loss)
    plt.title("Total Loss")
    plt.show()

    plt.plot(parse_train.iter, parse_train.text_loss)
    plt.plot(parse_train.iter, parse_train.ref_rpn_box_reg)
    plt.plot(parse_train.iter, parse_train.ref_objectness)
    plt.plot(parse_train.iter, parse_train.loss_rpn_box_reg)
    plt.plot(parse_train.iter, parse_train.loss_objectness)

    plt.legend(['Text Loss', 'RefExp Loss', 'RefExp Objectness', 'Segmentation Loss', "Segmentation Objectness"], loc='upper left')

    plt.show()

    print("Done")


if __name__ == "__main__":
    logfile = sys.argv[1]
    parse_log(logfile)