#!/bin/bash
#
# Execute web server and models

usage="Execute web server and models for local\n
  This will run on 127.0.0.1:5000\n

  usage:  run.sh [-H host] [-p port]\n
  \t-H: Database host\n
  \t-p: Database port to bind\n
"

host='127.0.0.2'
port=5432
worker=2

while getopts 'H:p:w:h' flag; do
  case "${flag}" in
    H) host="${OPTARG}" ;;
    p) port="${OPTARG}" ;;
    w) worker="${OPTARG}" ;;
    h) echo -e ${usage}
       exit
       ;;
    *) echo -e ${usage} >&2
       exit 1
       ;;
  esac
done

export DATABASE_HOST=${host}
export DATABASE_PORT=${port}
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD}

export PYTHONPATH=`pwd`

gunicorn -w ${worker} -b '127.0.0.1:5000' "run_webserver:main('local')" &
python3 -m main.models &
