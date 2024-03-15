from model import DenseNet

def DenseNet121(): return DenseNet(121)
def DenseNet169(): return DenseNet(169)
def DenseNet201(): return DenseNet(201)
def DenseNet264(): return DenseNet(264)

def get_model(DenseNet: str):
    DenseNets = {
        "DenseNet121": DenseNet121,
        "DenseNet169": DenseNet169,
        "DenseNet201": DenseNet201,
        "DenseNet264": DenseNet264
    }
    
    if DenseNet in DenseNets: 
        return DenseNets[DenseNet]()
    else:
        raise ValueError(f"Invalid DenseNet: {DenseNet}")