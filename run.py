from model import load_model, generate_caption


path = "models/15.02_15-06/pretrained_weights.h5"
caption_model = load_model(path)
pic_path = 'flickr8k/images/3006093003_c211737232.jpg'
images = [pic_path]
for img in images:
    pred_caption = generate_caption(img, caption_model)
    print(f"{img}| caption: {pred_caption}")
