# from model import get_caption_model, generate_caption
from cloned_get_model import get_caption_model, generate_caption
import os


# @st.cache(allow_output_mutation=True)
# def get_model():
#     return get_caption_model()
path = "models/15.02_15-06/pretrained_weights.h5"
caption_model = get_caption_model(path)
# images = [f"imgs/{f}" for f in os.listdir("imgs")]

# img_url = st.text_input(label='Enter Image URL')
pic_path = 'flickr8k/images/3006093003_c211737232.jpg'
images = [pic_path]
# img = np.array(img)
for img in images:
    pred_caption = generate_caption(img, caption_model)
    print(f"{img}| caption: {pred_caption}")
# st.write(pred_caption)
