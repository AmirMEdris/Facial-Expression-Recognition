# Emoition-classification-from-facial-expression
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
With the emergence of Augmented reality devices like hololens and the ever-increasing computation capabilities of mobile devices more and more machine learning projects are being made to deduce information from your surroundings.
the original dataset used for the emotion detection model can be found on Kaggle at the following link: https://www.kaggle.com/deadskull7/fer2013

## Repo Structure and Project Approach
 ### Facial Expression Recognition.pdf
 My presentation slides in pdf format

 ### Master_notebook.ipynb
 The Main notebook takes you through the all the steps I took to implement the the three different models that I used and using opencv to do realtime object detection using my models
 ### Modeling.ipynb
 This is the notebook I trained my models in on google cloud it has visuals for every model and some of my thoughts
 ### Functions.py
 pretty much a library of all the fucntions and imports I used to make things much less messy which just loads the architecture of the model that I used from Ashadullah Shawon.

 
![Image](https://github.com/AmirMEdris/Facial-Expression-Recognition/blob/main/Pics/add_picture_matplotlib_figure.png)
## Visualizing Classifications and modeling
neural networks and cnns in particular are known for being "black box" models due to the fact that they get results but arent very transparent about it. I used two methods to try and infer what the model thought was important. CNN’s contain dense layers much like their more basic counterparts, however, the convolutional layers that they use can each be output as its own individual picture showing the transformation that the model has that layer doing, doing this is called getting layer activations and all you really need to do is make a new model which is identical to the CNN you wish to visualize except it outputs before reaching the dense layers giving you the breakdown in a sense for the picture you would feed that model. The other method that I used, deep dream image generation, is a method by google where you feed an image into the model and calculate a loss function to represent how active the model is when looking at that image, and instead of decreasing the loss we do the opposite and so instead of the model tells you what the picture is you get something closer to what the model sees when it sees the input. I couldn’t generate these images for every model because I had only ever done this on much smaller scale models and didn’t realize they take a while to produce also my final model broke the deep dream visualization


![Image](https://github.com/AmirMEdris/Facial-Expression-Recognition/blob/main/Pics/Screen%20Shot%202020-11-12%20at%201.37.04%20PM.png)
The pics above are two pics that I chose one from the dataset, and one of me though the haar cascade classifier. Both pics are happy which and the model these pics were generated for whereof my initial model. The bottom row of pics are the results of the deep dream visualization and due to the resolution it is hard to discern but the model took the picture and outputted different features of faces at every location on the face. I chose two happy pictures because happy is the dominant class and is probably the easiest for us to judge the model’s interpretation. Since most of the faces are partial and we cant really say what expressions are present bringing in the other method will give us a better idea.

![Image](https://github.com/AmirMEdris/Facial-Expression-Recognition/blob/main/Pics/Screen%20Shot%202020-11-20%20at%2012.08.28%20PM.png)


This group of images shows the sum features extracted by each conv layer for 7 cases of the different classes. The normal way to view activations is filter by filter but instead of just that I added the filters together and got the features extract for the layer. Using the picture was very helpful in understanding what the models take away from each img was so I made a model evaluation function that would generate this among a confusion matrix for me to be able to look layer by layer and see how my changes to the model affected its learning. One way this helped me was in being able to add the right filter layer for mouths in the final model. Most of the models were struggling in iding lips but the 2 that didnt had bigger conv filters which helped me see that one wide conv filter was necessary. 
![Image](https://github.com/AmirMEdris/Facial-Expression-Recognition/blob/main/Pics/Screen%20Shot%202020-11-12%20at%202.59.05%20PM.png)

## Conclusions/Next Steps
- Improvements to the Haar Cascade Classifier can be made. It frequently struggles to find faces that are tilted more than 30 degrees and it often interprets stripes as faces. 
- A new supplemental dataset can be constructed in which the classes are changed to positive and negative using pose estimation. This data can be used to train an even more accurate model.
- Higher resolution pictures may provide more accurate results. The model currently has trouble with tasks such as finding lips and often misinterprets parts of an image due to low resolution. 
- A depth map would be an advantageous addition. Knowing the distance for each predicted body part can prove to be very useful.  
