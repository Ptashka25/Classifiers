import torchvision.transforms as T

preproccesing = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor()
    ]
)

def preprocess(img):
    return preproccesing(img)