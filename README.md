# TFG - Food Menu Recognition
Final project of the Computer Sciences BSc at the University of Barcelona.


## Requirements

The basic requirements of this projects are:

 - [Our version of Keras](https://github.com/MarcBS/keras) >= 1.2.3
 - [Multimodal Keras Wrapper](https://github.com/MarcBS/multimodal_keras_wrapper) >= 0.7


## Abstract
Food has become a very important aspect of our social activities. Since social
networks and websites like Yelp appeared, their users have started uploading photos
of their meals to the Internet. This factor leads to the development of food analysis
models and food recognition.
We propose a model to recognize the meal appearing in a picture from a list of menu
items (candidates dishes). Which could serve for the recognize the selected meal in a
restaurant. The system presented in this thesis does not need to train a new model
for every new restaurant in a real case scenario. It learns to identify the components
of an image and the relationship that they have with the name of the meal.
The system introduced in this work computes the similarity between an image and
a text sequence, which represents the name of the dish. The pictures are encoded
using a combination of Convolutional Neural Networks to reduce the input image.
While, the text is converted to a single vector applying a Long Short Term Memory
network. These two vectors are compared and optimized using a similarity function.
The similarity-based output is then used as a ranking algorithm for finding the most
probable item in a menu list.
According to the Ranking Loss metric, the results obtained by the model improve the
baseline by a 15%.
