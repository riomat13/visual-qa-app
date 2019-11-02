# 4 weeks challenge

This is a challenge to build Question Answering with Attention by images.
All images are from [COCO](http://cocodataset.org/), and quesion/answer dataset is from [VQA](https://visualqa.org/).

## Requirements
This model and app is running with these libraries in following versions.

```python
python==3.7
numpy==1.17.3
tensorflow==2.0
```

## 1st week - define the problem
Research related papers, models etc.
Data preprocessing.

### Data structure
Original json files. (These data should be downloaded from [website](https://visualqa.org/).)
Annotation: `data/annotations/v2_mscoco_train2014_annotations.json`(file name is modified)
Questions: `data/questions/v2_mscoco_train2014_questions.json`
Images: `data/train2014/COCO_train2014_{image_id:012d}.jpg`

### Structure of networks

#### 1. Image Encoding
In order to simplify the problem, Use *transfer learning* with `tf.keras.applications.MobileNet` for image encoding parts. It actually is not necessary to classify the images so that the last parameters will be used as input of decoder step. (It will be talked about decoder step later section.)

#### 2. Question type classification
Use *RNN* to capture sentences information and apply simple classification.

Apply *LSTM* to question sentence => output (class type is based on `https://github.com/GT-Vision-Lab/VQA/blob/master/QuestionTypes/mscoco_question_types.txt`)

Since this is pretty simple task, it can achieve more than *98%* accuracy in validation step with simple network and 30000 dataset(24000 for training and 6000 for validation).

#### 3. Question/Answering
In encoder step, parse question sentence with *RNN*, whereas use *RNN with Attention* in decoder step.
As target answer, load the answers data from dataset shown above. Each question has 10 ansewers and they will be used for training to be weighted importance since it may solve ambiguity.

### Question Type Classification
As initial step, we need to classify question types, and then pass the features (implement later) to decoder.
Prepared question types are used for classification. This has be easy since quesions come from first few words of a question sentence.
It can be done by checking first few words in sentences however, we can not be sure that questions are asked by following correct grammar nor withoud type therefore, build neural network to capture the feature and decide the most likely one is the question type to be used.
This model is going to be pre-trained and use the weights at later model(decoding model).

```python
python run_questiontype_classification.py

# if you want to test by your own sentence
python run_questiontype_classification.py -i
```

### Web Application
In order to access model, web API will be implemented, which is built with `flask`.
This is for serving models created, therefore it may not be developed for UI, but just for API.

## 2nd week - build minimum model
Build minimum vaiable product(MVP) model. Send data by API and return the results by `json` format.

## 3rd week - deploy with simple structure
Make the model robust. Build interface to serve the model.

## 4th week - scaling
Scaling and deploy.

## Additional weeks
Add more features such as rich UI or additional pages in web pages.

## Reference:
- [COCO](http://cocodataset.org/)
- [VQA](https://visualqa.org/)
- [Image Captioning(tensorflow official)](https://www.tensorflow.org/tutorials/text/image_captioning)
- A.G. Howard et al, MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, 2017 [[link]](https://arxiv.org/abs/1704.04861)
