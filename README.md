# Emotion-classification-from-facial-expression
By Amir Edris 
## Table of Contents
1. Project Overview
1. Business Case 
1. Repo Structure and Project Approach
1. Modeling 
1. Final Modeling and Analysis
1. Visualizing Classifications
1. Conclusions/ Next Steps
## Business Case 
Social-Emotional Agnosia is the inability to perceive nonverbal cues. Experts say that 70-93% of communication is non-verbal meaning that the vast majority of information being conveyed in a conversation resides in visual cues such as body language or facial expression. As a result, people who suffer from Social-Emotional Agnosia often have to learn to identify these social cues on their own which you can imagine isn’t as easy as it sounds. Using object detection and image classification techniques I believe we can provide a real-time classification of at least some of these nonverbal cues and provide aid to those who struggle with these skills. 
## Project Overview
With the imergence of Augmented reality devices like hololens and the ever increasing computation capabilities of mobile devices more and more machine learning projects are being made to deduce information from your surroundings.  
the original dataset used for the emotion detection model can be found on Kaggle at the following link: https://www.kaggle.com/deadskull7/fer2013
### class imbalance
![Image](https://github.com/AmirMEdris/Facial-Expression-Recognition/blob/main/Pics/Screen%20Shot%202020-11-20%20at%2012.08.52%20PM.png)
## Repo Structure and Project Approach

 ### Master_notebook.ipynb
 The Main notebook takes you through the all the steps I took to implement the the three different models that I used
 ### Model Visualizations.ipynb
  - the visualizations that I make use feature extraction from the cnn model used in the main notebook by outputting the result of each conv layer prior to the dense ones
  - the next set of visualizations that I plan to do are to use the google deep dream gradient ascent methood that to maximize loss in a picture for each layer of the cnn to get a nice representation of what the model is looking for by generating the picture that would excite the model the most.
 ### Functions.py
 - initially both of the previous notebooks were a complete mess so I made a new file named functions where I essentially dumped every function definition and every module import and in there included functions like "my_model" which just loads the arcitecture of the model that I used from Ashadullah Shawon.
 

## Visualizing Classifications and modeling
neural networks and cnns in particular are known for being "black box" models due to the fact that they get results but arent very transparent about it. I used two methods to try and infer what the model thought was important. CNNs contain dense layers much like their more basic counterparts, however, the convolutional layers that they use can each be output as its own individual picture showing the transformation that the model has that layer doing, doing this is called getting layer activations and all you really need to do is make a new model which is identical to the cnn you wish to visualize except it outputs before reaching the dense layers giving you the breakdown in a sense for the picture you would feed that model. The other methood that I used, deep dream image generation, is a methood by google where you feed an image into the model and calculate a loss function to represent how active the model is when looking at that image and instead of decreasing the loss we do the oppisite and so instead of the model telling you what the picture is you get something closer to what the model sees when it sees the input. I couldnt generate these images for everymodel because I had only ever done this on much smaller scale models and didnt realise they take a while to produce also my final model broke the deep dream visualization

![Image](https://github.com/AmirMEdris/Facial-Expression-Recognition/blob/main/Pics/Screen%20Shot%202020-11-12%20at%201.37.04%20PM.png)
The pics above are two pics that I chose one from the dataset, and one of me though the haar cascade clasifier. Both pics are happy which and the model these pics where generated for where of my initial model. The bottom row of pics are the results of the deep dream visualization and due to the resolution it is hard to discern but the model took the picture and outputed different features of faces at every location on the face. I chose two happy pictures because happy is  the dominant class and is probably the easiest for us to judge the models interpretation. Since most of the faces are partial and we cant really say what expressions are present bringing in the other methood will give us a better idea.

![Image](https://github.com/AmirMEdris/Facial-Expression-Recognition/blob/main/Pics/Screen%20Shot%202020-11-20%20at%2012.08.28%20PM.png)


the first group of images in orange are the avg conv layer output for 7 cases of the different classes. this is why I dont really like using activations as much because it literally generates 500-1000 photos as opposed to one with all the features included. Any way the green group of pictures are heatmaps over what the filters thought were inportant in their classification and if you look the eyes tend to draw much attention to the model even though they are fairly normal. While the filters do acknolege the smile it seems to be overshawdowed by the eye. Multiple different cases like this are what led me to think that this model wasnt deep enough and was maybe picking up on something suttle but important like microexpressions around the eyes but just poorly.

With a 60% accuracy on the val set the model improved by a 4% margin even though it is probably triple the size intrestingly though it did amazing on the minority class disgust 
![Image](https://github.com/AmirMEdris/Facial-Expression-Recognition/blob/main/Pics/Screen%20Shot%202020-11-12%20at%202.59.05%20PM.png)

## Conclusions/Next Steps
There is so many directions to go in from this point. I think that the Haar Cascade Classifier can be improved because most of the time it struggles to find faces that are tilted more than 30 degrees also it frequently thinks that stripes are faces. I think you can also change the classes to positive and negative then make a new set of data in conjunction with the pose estimator and make a model that classifies positive vs negative pose for a person and then be able to more accurately predict emotion using this new info. I’m also tempted to say a dataset of higher resolution pictures may provide more accurate results especially due to the fact that I had to downscale most of the images to try to replicate the 48,48 res but most of the time there seemed to be a difference, however, for realtime image classification, I think that would probably half the already low fps. Finally, I think that a depth map would also be a good thing to add to be able to know if the distance for each predicted body part.
