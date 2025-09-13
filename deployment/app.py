from fastai.vision.all import *
import gradio as gr
from PIL import Image

model = load_learner('models/landmark_recognizer_model_v1.pkl')

landmark_labels = model.dls.vocab

def recognize_image(image):
    img= image.resize((192, 192))
    pred, idx, probs = model.predict(img)
    return dict(zip(landmark_labels, map(float, probs)))

image = gr.Image(type="pil")
label = gr.Label(num_top_classes=5)
examples = [
    './test_images/unknown01.jpeg',
    './test_images/unknown02.jpg',
    './test_images/unknown03.jpg',
    './test_images/unknown04.jpg',
    './test_images/unknown05.jpeg',
    './test_images/unknown06.jpg'
    ]
iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False,debug=True)