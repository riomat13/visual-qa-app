# 4 weeks challenge

This is a challenge to build Question Answering with Attention by images.
All images are from [COCO](http://cocodataset.org/), and quesion/answer dataset is from [VQA](https://visualqa.org/).

Note) In this project, I use one laptop, and *Jetson TX2* for training model, another PC is used to develop web API. Therefore, I can not make enough time to run and test models a lot and simultaneously.

## Requirements
This model and app is running with these libraries in following versions.
See more detail in `requirements.txt`.

```python
python==3.7.5

flask==1.1.1
Flask-WTF==0.14.2
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
matplotlib==3.1.2
numpy==1.17.3
Pillow==6.2.1
SQLAlchemy==1.3.10
tensorflow==2.0.0

# for training with Jetson TX2
python==3.6.8
tensorflow-gpu==1.14.0+nv19.9
```

### Database settings (Local)
To make tasks simple, used `postgresql` on `docker`.
Following is the simple steps to set up.
```bash
# create postgresql container
user@user-pc$ docker run -ti --name some-name -p 5432:5432 -e POSTGRES_PASSWORD=any-password -d postgres

# login to container
docker exec -ti some-name bash
root@5f...:/# psql -U postgres

# manually set up database
postgres=# CREATE DATABASE db_name;
postgres=# \q

root@5f...:/# exit;

# login to postgres in container from local machine
user@user-pc$ psql -h container-host -p 5432 -U postgres
```

## 1st week - define the problem
This is *Visual Question Answering* problem. Due to limited resources and time and this includes deploying as simple web app, multiple models will be built and may not solve all question types. Therefore, it might be "Can't answer the question" as the answer. If I have additional time to develop this, may change structure but for now, I will make progress with this way. Followings are the some details about the steps.

### Data structure
Original json files. (These data should be downloaded from [website](https://visualqa.org/).)

- Annotation: `data/annotations/v2_mscoco_train2014_annotations.json`(file name is modified)

- Questions: `data/questions/v2_mscoco_train2014_questions.json`

- Images: `data/train/processed/{image_id:06d}.npy` (Encoded by *MobileNet* to save process time in training)

### Structure of networks

#### 1. Image Encoding
In order to simplify the problem, Use *transfer learning* with `tf.keras.applications.MobileNet` for image encoding parts, which is based on *MobileNet*[4]. It actually is not necessary to classify the images so that the last parameters will be used as input of decoder step. (It will be talked about decoder step later section.)

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

### Answering Yes/No
As the easier problem to answer yes or no than to do with sentence, I thought it should be the good place to start.
However, since answering given question based on image, it has to encode both question and image and process those data and make prediction.

After a bunch of trials & errors, I could get better result applying *Attention* to both image and sentence. I could see how weighted the words in sentence. (Surely padded words has low importance, but some actual words have high ones and it looks working. However, all existing words still have higer values in many times so that it is need to more improvement)

This problem will be the base to build others. Therefore carefully chose networks. (Technically, many of networks failed to learn, and only work with *Attention* with additive way[5].)

```python
python run_yes_no_answering_model.py

# for tensorflow 1.14 (this is only used for developing model)
python run_yes_no_answering_model.1.14.py
```

### Answering to 'What is/are...' type question
This should be one of the most common types of question about images, so that this is the second priority to build model. This is going to be used *Attention* such like *Yes/No* answering but answer is decoded by *seq2seq* model. That is, pass the beggining word, and predict words one by one and eventually generate the answer.


### Web Application
In order to access model, web API will be implemented, which is built with `flask`.
This is for serving models created, therefore it may not be developed for UI, but just for API.

## 2nd week - build minimum model
In order to make simplify the model, I decided to split the problems into smaller ones.
For instance, firstly, classify the problem such as "this is closed/opened question", or 
this question asks 'what' or 'where' or others" and so on.
And then, build other models based on the results.

To develop simplest models, I started from *closed* question such as *'yes'* or *'no'*. Although it is simple classification problem, it is hard to predict since the answers are based on given images. Therefore, *Attention Mechanism* is used to get important information from both sentences(questions) and images.

As I mentioned, it is hard to get good result, therefore it took so much time to build models and also there is still *high variance*, which could not be resolved by simple regularization nor adding more data. There may be better ways to resolve this but since this is limited time project, I moved on to next step.

## 3rd week - deploy with simple structure
Developing simple web UI to upload an image and ask a question. I have used `flask` for it, and it can serve the result (for now(11/14), it can only return question type, because closed question model is still running in training).

Additionally, one of the most frequent question type is *'what is/are ...'*
so that started to build models.
The base structure is quite similar to closed question model.
The only difference is that in decoding step, *seq-to-seq* like model
so that the previous word is passed and predict the next word
(initial word is '\<BOS\>' to trigger the sequence generation and '\<EOS\>' is the end of sentence).

## 4th week
Built model for yes/no answering, what, and why, so that it can run these predictions.
Otherwise, it returns 'Could not answer the question'.

Since it takes so long time to train models, added some particular models in this time and will add other models later.
In order to deploy model with simple and easy way, `heroku` is used this time.
The branch name is `heroku` and `Procfile` is added to this branch.
This only has two simple servers, which are `web` to run flask app and
`workers` to process images to avoid consuming memory space, which are not included this repository since it is `heroku` specific.

Unfortunately, I spent a lot of time to build and test models,
so that I could not set up more scalable app using *MySQL* or *PostgreSQL* for database,
and *SQLite* is used instead which has many limitation but as small app it works fine.
This will be an additional task after 4 weeks.

Currently it runs on `https://visual-qa.herokuapp.com/` with simple interface.
However it uses free acount, so that it works as *synchronous* processing,
thus, it may take a while to get prediction result.

## Future works
Add more features such as rich UI or additional pages in web pages. What done in this project will be the baseline, and try to improve the result with the same dataset.

This time the model is separated based on question type.
However it works well with one same structure of model, even though the answer is vary such as yes/no, one word, or sentence.

As the model structure, *Attention*s and *GRU*s are used this time but
they are structured sequentially, that is, it can not fully utilize *GPU* strength,
which is vectorization.

- Processing uploaded image for later use and compressing data size
- Update models to improve performance both in accuracy and calculation time
  Such as Memory Network, Attention without sequence

## Reference:
[1] [COCO](http://cocodataset.org/)

[2] [VQA](https://visualqa.org/)

[3] [Image Captioning(tensorflow official)](https://www.tensorflow.org/tutorials/text/image_captioning)

[4] A.G. Howard et al., MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, 2017 [[link]](https://arxiv.org/abs/1704.04861)

[5] D. Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate, 2014 [[link]](https://arxiv.org/abs/1409.0473)