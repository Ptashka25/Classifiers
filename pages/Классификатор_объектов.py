import streamlit as st
from PIL import Image
from torchvision.models import inception_v3
from pages.model_2.preproccesing import preprocess
import json


@st.cache_resource()
def load_model():
    model = inception_v3(pretrained=True)

    model.eval()

    return model

model = load_model()

image = st.file_uploader('Загрузите фотографию')


def predict(img):
    img = preprocess(img)
    img = img.unsqueeze(0)
    pred = model(img)
    return pred


labels = json.load(open('C:\ds_bootcamp\Проекты\Classifications\pages\model_1\imagenet_class_index.json'))
decode = lambda x: labels[str(x)][1]

if image:
    image = Image.open(image)
    prediction = predict(image)
    st.markdown(f"<h1 style='text-align: center;'>{decode(prediction.argmax(dim=1).item())}</h1>", unsafe_allow_html=True)
    st.image(image, caption="", use_column_width=True)