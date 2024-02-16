# from the vid
# inception_v3 = tf.keras.applications.InceptionV3(
#     weights='imagenet',
#     include_top=False,
# )
#
# # inception_v3.summary()
#
# def load_img(img_path):
#     img = tf.io.read_file(img_path)
#     img = tf.io.decode_jpeg(img, channels=3)
#     img = tf.keras.layers.Resizing(299, 299)(img)
#     img = img / 255. # normalize
#     return img
#
# def get_feature_vector(img_path):
#     img = load_img(img_path)
#     img = tf.expand_dims(img, axis=0)
#     feature_vector = inception_v3(img)
#     return img, feature_vector
#
# img, feature_vector = get_feature_vector('flickr8k/images/10815824_2997e03d76.jpg')
#
# plt.imshow(np.squeeze(img, axis=0))
# plt.axis('off')
# plt.show()
#
# print(f"input image size: {img.shape}")
# print(f"feature vector size: {feature_vector.shape}")
