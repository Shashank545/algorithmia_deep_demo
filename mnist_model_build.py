import Algorithmia
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model



# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages

    
def mnist_model_inference(input):
    # load the image
	img = load_img(input, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	
	model = load_model('mnist_2e.h5')
	# predict the class
	digit = model.predict_classes(img)
	print(digit[0])
	return digit[0]