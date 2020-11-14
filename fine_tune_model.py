from keras.models import load_model

mode_name = "model_muti_v2.h5"


model = load_model(mode_name)

model.summary()