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
## Project Overview
It said that, when it comes to spoken language, the majority of sentiment can be deduced from visual cues such as body language and facial expression. This means that there is a vast amount of information that can be deduced from images. Many people even invest time into studying body language for a vast majority of reasons from being able to get messages across to people to improving social skills. Regardless of what angle you look at it from many people would agree that at least being able to document a reasonably accurate emotion detection model would be very benefical to many people.
the original dataset used for the emotion detection model can be found on Kaggle at the following link: https://www.kaggle.com/deadskull7/fer2013
## Business Case 
Imagine you are pitching a business idea on the very popular show shark tank. Lets say your idea is something in the medical field. Now alot of people that watch shark tank will know that mark cuban tends to be skeptical of pitches in the medical field particularly if the persons ideas cannot be backed by their own research. So if you are doing your medical pitch on shark tank and the model gives you the information that mark has shown disgust anger and sadness while you were speaking you can now readjust who you are pitching too since its clear that you wont be getting an offer from him any more you can cater to the remaining sharks. While this might be a specific example I think this conveys the oppourtunity that a model like this creates for people who have to convey ideas.

## Repo Structure and Project Approach
  When I started this project I was mainly interested in seeing how much progress I could make so before training models I wanted to see If I could get to the point of sucessfully predicting pose and emotion on multiple people in realtime. With this goal in mind I decided to temporarily use the weights from someone elses model for the image dataset and see how far into my goal I can get. Initially I used open cv to capture my cameras live feed in a while loop. then every frame I would lower the resolution into the one that the model accepts of 48 by 48 and grayscale it which was terrible because this methood took a 1080p frame and downscaled the entire thing meaning that the entire 1080p picture would one look horrible and two the entire picture which might not even have a face at all will be treated as if its one picture of one face. The first fix that I attempted was to only downscale a subset of the 1080p picture. This helped my problem because now I was downscaling a 300 by 300 square to 48 by 48 instead of an hd rectangle. the features of the face were easier to see but the problem of the entire picture being treated like a face regardless of how many faces are actually present remained. I started to think about how to go about making a model that detects faces present in a picture. This led me to read about different methoods on how to do this then finally I stumbled upon a model which is even included in open cv called a Haar Cascade Classifier which was first used around 2001 and is essentially a model that knows the group of features present in a face and can tell you where in a picture a face is located even if there are multiple. So this fixed my problem and making a function that used opencvs version of this model for facial detection and returned the face images and the faces coords is where this project started to come together. next I used the tf pose module for the mobile net module which is a cnn that can predict the locations of different body parts which is a much more complicated model than the harrcascade because this model consists of multiple different cnns and has to make a ton of different predictions instead of is this group of pixels a face. From this point ive spent time trying to optimize the different fucntions ive had to make to make this understandable and troublshooting a ton of different shape errors also working on feature extraction and pattern recognition using deep dream and modeling layer output.
 ### Master_notebook.ipynb
  - The model that I used for for face detection is a HaarCascadeClassifier from opencv
  - The weights for the cnn that makes emotion prediction Is from kaggle user Ashadullah Shawon
  - Pose estimation came from the mobilenet v2 model which I am using from the tf-pose module which is a subset of the famous cmu pose estimation model
 ### Model interp.ipynb
  - the visualizations that I make use feature extraction from the cnn model used in the main notebook by outputting the result of each conv layer prior to the dense ones
  - the next set of visualizations that I plan to do are to use the google deep dream gradient ascent methood that to maximize loss in a picture for each layer of the cnn to get a nice representation of what the model is looking for by generating the picture that would excite the model the most.
 ### Functions.py
 - initially both of the previous notebooks were a complete mess so I made a new file named functions where I essentially dumped every function definition and every module import and in there included functions like "my_model" which just loads the arcitecture of the model that I used from Ashadullah Shawon.
## Modeling 
-incomplete
## Final Modeling and Analysis 
-incomplete
## Visualizing Classifications 
Now that the underlying framework is put into place and fully functional I am going to start to work on improving the models accuracy and finally changing the arcetecture from the inital borowed one. also going to add a baseline model wich is long overdue.


## Conclusions/Next Steps 
As of right now after working on the iterative modelling process and getting a model that Im more confident in I plan on generating a new dataset that takes the pose estimation information and the predicted facial expression then making a new model that can take both someones facial expression and their pose( which would be similar but I havent thought about how to effectively do this step yet). I also whant to add hand gesture recognition, speech recognition, and sentiment from speech but most of this probably wont be accomplishable by next week and eventually you will run into fps issues going down this path and for what should be a relatively light weight model thatll be too much.

