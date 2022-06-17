from data . dataloader import MNIST_Loader
import yaml
import cv2


def load_yml(yml_path):
    with open(yml_path) as tyaml:
        yml = yaml.safe_load(tyaml)
        return yml

def main():
    config  = load_yml('yml/train.yml')
    loader = MNIST_Loader(config)

    data,label = loader.next()

    print(data)
        

if __name__ == '__main__':
    main()

