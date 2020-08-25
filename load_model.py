from keras import models


def create_model():
    new_model = models.load_model('malaria.h5')
    return new_model

if __name__ =='__main__':
    model = create_model()
