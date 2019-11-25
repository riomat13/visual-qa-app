# Visual Question Answering App

This is a challenge to build Question Answering with Attention by images.
All images are from [COCO](http://cocodataset.org/), and quesion/answer dataset is from [VQA](https://visualqa.org/).

(Originally started this as 4-week challenge. The note about it is in `note.md`.)

## Requirements
Detail is in `requirements.txt` but basically followings are used:
```
python==3.7.5

flask==1.1.1
Flask-WTF==0.14.2
gunicorn==19.9.0  # do not work with gunicorn 20.0
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
matplotlib==3.1.2
numpy==1.17.3
Pillow==6.2.1
SQLAlchemy==1.3.10
tensorflow==2.0.0
```

## Package Structure
```
.
|─ checkpoints        # store all model data such as weights
|─ data               # store data used for model
|─ main
|   |─ alembinc.ini   # database settings
|   |─ migrations     # migration data
|   |─ mixins
|   |─ models         # ML models to predict
|   |─ orm            # orm related such as db models, session, engine
|   |─ settings       # configuration for the app
|   |─ utils
|   └─ web            # simple web app
|       |─ api        # web api
|       |─ static     # static files
|       |   |─ css
|       |   |─ js
|       |   └─ media  # store images
|       |─ static     # static files
|       └─ templates  # html files
|─ tests
|─ note.md            # note about 4 weeks challenge
|─ README.md
|─ requirements.txt
|─ run_*.py           # scripts for training with tf.2.0
|─ run_*.1.14.py      # scripts for training with tf.1.14 for Jetson
└─ run.sh             # execute web server and model server in background
└─ run_webserver.py   # execute web server
```

## Execute App
```bash
# running on local machine
export DATABASE_HOST='127.0.0.1'  # database host
export DATABASE_PORT=5432         # port to bind (this is for postgresql)

# if necessary set password as well
export POSTGRES_PASSWORD='somepassword'

gunicorn -b 127.0.0.1:5000 "run_webserver:main('local')"

# running model server
python -m main.models
```