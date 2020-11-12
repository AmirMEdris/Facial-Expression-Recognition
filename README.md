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
With the imergence of Augmented reality devices like hololens and the ever increasing computation capabilities of mobile devices means that it is becoming more and more possible to use machine learning to to can start to play  , the possibily of  Thus, Parasite counts are used to diagnose malaria correctly. However, the manual counting of parasites varies depending on the skill and expertise of the microscopist. Depending on the conditions and treament of these scientisits, accurate diagnosis can vary. Thus, the goal of this project is to create a highly accurate image classification model that can quickly and effictively diagnose Malaria.The model will be trained on a kaggle dataset of approximatly 28,000 photos of infected and uninfected cells. The original dataset cvan be found on Kaggle at the following link: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria


the original dataset used for the emotion detection model can be found on Kaggle at the following link: https://www.kaggle.com/deadskull7/fer2013

## Repo Structure and Project Approach

 ### Master_notebook.ipynb
 The Main notebook takes you through the all the steps I took to implement the the three different models that I used
 ### Model Visualizations.ipynb
  - the visualizations that I make use feature extraction from the cnn model used in the main notebook by outputting the result of each conv layer prior to the dense ones
  - the next set of visualizations that I plan to do are to use the google deep dream gradient ascent methood that to maximize loss in a picture for each layer of the cnn to get a nice representation of what the model is looking for by generating the picture that would excite the model the most.
 ### Functions.py
 - initially both of the previous notebooks were a complete mess so I made a new file named functions where I essentially dumped every function definition and every module import and in there included functions like "my_model" which just loads the arcitecture of the model that I used from Ashadullah Shawon.
## Modeling 

## Final Modeling and Analysis 

## Visualizing Classifications 

 ### Master_notebook.ipynb
  - Import all necessary libraries and modules to use and manipulate image data as well as build  models
  - For ease of use, multiple functions are declared at the top of the notebook
  - Look at Regular and Infected Cells to look for visual differences
  - Look at dataset's class distribtuions and use to build a dummy classifier model
  - Prepare both complicated and simple CNN models to see how well they perform classfications, try different hyperparamters to optimize performance.
  - Compare Models' performances on evaluation metrics and select the best performing model
  - Select Final model and test for overfitting, correct for overfitting accordingly
  - Look at final model's misclassifications and analyze those
  - Present Conclusions,recommendations, next steps
 ### Model interp.ipynb
 - contains code aand functions to visualize what the model is seeing at different layers of the netework
 
## Modeling 
For Modeling, We believed that testing four types of models and consistently altering hyper parameters would best allow us to find the ideal model. We first created a dummy model that would predict the dominant class. However, as you can see above, there was no numerically dominant class. Thus, We chose the infected class as the dominant to minimize false negatives. This model achieved an accuracy of exactly 50%. We also decided to create a simpler CNN model with less convultional and dense layers. This model achieved an accuracy of 93% on the validation set. We then decided to create a more complex CNN model to see how well it would perform. The complex CNN model achieved an accuracy of about 95% on the validation set. We then decided to use Transfer Learning to determine whether that could further improve performance. We used Google's prebuilt Model InceptionV3 which is trained on large amounts of medical data.  We acheived an accuracy of 92% on the validation set for that model.

## Final Modeling and Analysis 
Due to it outperforming all other models, We selected the Complex CNN model as our final model. Achieving 95% Accuracy on both our test sets and validation sets was certainly impressive. Also, this model achieved a recall of .96 on the infected class, We were very satisified with this result because it proved that we are effectively limiting our false negatives as well.

![Image](https://github.com/Scogs25/Malaria_Classification_Project/blob/main/pngs/Final_model_Accuraccy_curve.png)
![Image](https://github.com/Scogs25/Malaria_Classification_Project/blob/main/pngs/Final_model_Confusion_Matrix.png)

As can be seen above, this model accuractly classifies whether or not a cell is infected and it also minimizes incorrect classifications. Thus, we can feel confident this model can be highly effective as a diagnosis tool.

## Visualizing Classifications 
neural networks and cnns in particular are known for being "black box" models due to the fact that they get results but arent very transparent about it. I used two methods to try and infer what the model thought was important. CNNs contain dense layers much like their more basic counterparts, however, the convolutional layers that they use can each be output as its own individual picture showing the transformation that the model has that layer doing, doing this is called getting layer activations and all you really need to do is make a new model which is identical to the cnn you wish to visualize except it outputs before reaching the dense layers giving you the breakdown in a sense for the picture you would feed that model. The other methood that I used, deep dream image generation, is a methood by google where you feed an image into the model and calculate a loss function to represent how active the model is when looking at that image and instead of decreasing the loss we do the oppisite and so instead of the model telling you what the picture is you get something closer to what the model sees when it sees the input. I couldnt generate these images for everymodel because I had only ever done this on much smaller scale models and didnt realise they take a while to produce also my final model broke the deep dream visualization

![Image](https://github.com/Scogs25/Malaria_Classification_Project/blob/main/pngs/deepdream.png)

## Conclusions/Next Steps 
There are so many direction to go in from this point. I think that the Haar Cascade Classifier can be improved because most of the time it struggles to find faces that are tilted more than 30 degrees also it frequently thinks that stripes are faces. I think you can also change the classes  to positive and negative than make a new set of data in conjunction with the pose estimator and make a model that classifies positve vs negative pose for a person and then be able to more acuratly predict emotion using this new info. Im also tempted to say a dataset of higher resolution pictures may provide more accurate results especially due to the fact that I had to downscale most of the images to try to replicate the 48,48 res but most of the time there seemed to be a difference, however for realtime image classification I think that would probably half the already low fps. Finally I think that a depth map would also be a good thing to add to be able to know if distance for each predicted body part. 
