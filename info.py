from torchinfo import summary

from utils import get_model

def main():
    densenets = ["DenseNet121", "DenseNet169", "DenseNet201", "DenseNet264"]
    for densenet in densenets:
        model = get_model(densenet)
        print(densenet)
        summary(model, input_size=[1, 3, 224, 224])
        print("\n\n")

if __name__ == "__main__":
    main()