import os
import random
import argparse

random.seed(0)
parser = argparse.ArgumentParser("Generate SubImageNet", add_help=False)
parser.add_argument("--data-path", default="../data/imagenet", type=str, help="dataset path")
parser.add_argument("--output-path", default="../data/subImageNet", type=str, help="output path for subImageNet")
args = parser.parse_args()

data_path = args.data_path
output_path = args.output_path
ImageNet_train_path = os.path.join(data_path, "train")
subImageNet_name = "subImageNet"
class_idx_txt_path = os.path.join(output_path, subImageNet_name)

# train
classes = sorted(os.listdir(ImageNet_train_path))
if not os.path.exists(os.path.join(output_path, subImageNet_name)):
    os.mkdir(os.path.join(output_path, subImageNet_name))

subImageNet = dict()
with open(os.path.join(class_idx_txt_path, "subimages_list.txt"), "w") as f:
    subImageNet_class = classes
    for iclass in subImageNet_class:
        class_path = os.path.join(ImageNet_train_path, iclass)
        if not os.path.exists(os.path.join(output_path, subImageNet_name, iclass)):
            os.mkdir(os.path.join(output_path, subImageNet_name, iclass))
        subImages = random.sample(sorted(os.listdir(class_path)), 100)
        f.write("{}\n".format(subImages))
        subImageNet[iclass] = subImages
        for image in subImages:
            raw_path = os.path.join(ImageNet_train_path, iclass, image)
            new_ipath = os.path.join(output_path, subImageNet_name, iclass, image)
            os.system("cp {} {}".format(raw_path, new_ipath))

sub_classes = sorted(subImageNet.keys())
with open(os.path.join(class_idx_txt_path, "info.txt"), "w") as f:
    class_idx = 0
    for key in sub_classes:
        images = sorted((subImageNet[key]))
        f.write("{}\n".format(key))
        class_idx = class_idx + 1
