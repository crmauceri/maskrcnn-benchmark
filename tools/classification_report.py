import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.imports import import_file

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Function from scikit tutorial https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax

# Most of this code is from demos/predictor.py

def main(cfg):

    # Load saved network
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    model.eval()

    super_gt = []
    fine_gt = []
    super_pred = []
    fine_pred = []

    categories = None
    super_categories = None
    fine_categories = None
    continuous_index = None

    # Split=False is Test set
    data_loaders = make_data_loader(cfg, split=False, is_distributed=False)
    for dataset_index, data_loader in enumerate(data_loaders):
        dataset = data_loader.dataset

        if categories is None:
            categories = dataset.coco.cats
            continuous_index = dataset.contiguous_category_id_to_json_id
            super_categories = list(set([categories[value]['supercategory'] for key,value in
                                                continuous_index.items()]))
            fine_categories = ['unknown']
            fine_categories.extend([categories[value]['name'] for key, value in
                                         continuous_index.items()])


        for index, instance_t in tqdm(enumerate(data_loader), desc=cfg.DATASETS.TEST[dataset_index]):
            try:
                instance, target = model.prepare_instance(instance_t[0], device=cfg.MODEL.DEVICE)
                with torch.no_grad():
                    prediction = model(instance_t[0], device=cfg.MODEL.DEVICE)

                super_gt.append(super_categories.index(categories[continuous_index[target.item()]]['supercategory']))
                fine_gt.append(continuous_index[target.item()])

                _, pred_ind = prediction[:,-1,:].max(1)
                super_pred.append(super_categories.index(categories[continuous_index[pred_ind.item()]]['supercategory']))
                fine_pred.append(continuous_index[pred_ind.item()])
            except FileNotFoundError as e:
                print(e)

        fig1, ax1 = plot_confusion_matrix(super_gt, super_pred, classes=np.array(super_categories), title="Supercategory confusion", normalize=True)
        fig2, ax2 = plot_confusion_matrix(fine_gt, fine_pred, classes=np.array(fine_categories), title="Fine-grained confusion", normalize=True)

        fig1.savefig('{}/{}_supercategory_confusion.pdf'.format(cfg.OUTPUT_DIR, cfg.DATASETS.TEST[dataset_index]), bbox_inches='tight')
        fig2.savefig('{}/{}_fine_confusion.pdf'.format(cfg.OUTPUT_DIR, cfg.DATASETS.TEST[dataset_index]), bbox_inches='tight')

        super_report = classification_report(super_gt, super_pred, target_names=np.array(super_categories)[unique_labels(super_gt, super_pred)])
        fine_report = classification_report(fine_gt, fine_pred, target_names=np.array(fine_categories)[unique_labels(fine_gt, fine_pred)])

        print(super_report)
        print(fine_report)

        with open('{}/{}_classification_report.txt'.format(cfg.OUTPUT_DIR, cfg.DATASETS.TEST[dataset_index]), 'w') as f:
            f.write(super_report)
            f.write('\n')
            f.write(fine_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/LSTM_classification_experiment.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.freeze()

    main(cfg)