from keras.models import model_from_json
from keras.models import load_model
from random import randint
import cv2
import numpy as np

# Load the Model from Json File
#json_file = open('./model_test_3/model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)

# Load the weights
#loaded_model.load_weights("./model_test_3/model.h5")
# print("Loaded model from disk")

loaded_model = load_model('model.h5')

# Compile the model
loaded_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

guess = randint(1, 12)

# Load the image
image = cv2.imread('./dataset/single_prediction/guess_'+str(guess)+'.jpg')
# image = cv2.imread('./dataset/single_prediction/guess_5.jpg')
# image = cv2.imread('./dataset/test_set/cats/cat.4025.jpg')
# image = cv2.imread('./data/train/cats/cat.30.jpg')
print(guess)
cv2.imshow("Input Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#image = cv2.resize(image, (150, 150))
#image = image.reshape(1, 150, 150, 3)

image = cv2.resize(image, (64, 64))
image = image.reshape(1, 64, 64, 3)


# Predict to which class your input image has been classified
result = loaded_model.predict_classes(image)
if result[0][0] == 1:
    print("This is a Dog")
else:
    print("This is a Cat")
