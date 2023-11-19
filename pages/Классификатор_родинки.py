import streamlit as st
from PIL import Image
import torch
from torchvision.models import resnet50
from pages.model_2.preproccesing import preprocess

@st.cache_resource()
def load_model():
    model = resnet50()
    model.fc = torch.nn.Linear(in_features=2048, out_features=1)
    weights = torch.load('C:\ds_bootcamp\Проекты\Classifications\pages\model_3\savemodel3.pt')

    model.load_state_dict(weights)

    model.eval()

    return model

model = load_model()

image = st.file_uploader('Загрузите фотографию родинки')


def predict(img):
    img = preprocess(img)
    img = img.unsqueeze(0)
    pred = model(img)
    return pred


dict_ = {0: 'Доброкачественная',
        1:'Злокачественная'}

if image:
    image = Image.open(image)
    prediction = predict(image)
    st.markdown(f"<h1 style='text-align: center;'>{dict_[torch.sigmoid(prediction).round().item()]}</h1>", unsafe_allow_html=True)
    st.image(image, caption="", use_column_width=True)