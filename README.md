# 4 weeks challenge

This is a challenge to build Question Answering with Attention by images.
All images are from [COCO](http://cocodataset.org/), and quesion/answer dataset is from [VQA](https://visualqa.org/).

## Requirements
This model and app is running with these libraries. (haven't tested with other versions)

```python
python==3.7
tensorflow==2.0
flask==1.1.1
SQLAlchemy==1.3.10
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

Apply *LSTM* to question sentence => output (class type is based on `data/QeustionTypes/mscoco_question_types.txt`)

#### 3. Question/Answering
In encoder step, parse question sentence with *RNN*, whereas use *RNN with Attention* in decoder step.
As target answer, use the answers data from dataset shown above, and 


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
