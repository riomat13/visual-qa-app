#!/bin/bash

TESTS_LOG=./tests/logs/tests.log
COVERAGE_LOG=./tests/logs/coverage.log

function clean_files () {

  # make files empty
  cat /dev/null > $TESTS_LOG
  cat /dev/null > $COVERAGE_LOG

}

function run_coverage () {
  start_time=$(date -u +%s.%M)
  count=1

  for file in $(find ./tests -type f -name 'test_*.py'); do

    file=$(echo ${file} | cut -d'/' -f2-)

    echo "  ${count}. ${file}" >> $TESTS_LOG
    printf "%4.4s. %s\n" ${count} ${file} >> $TESTS_LOG
    echo '======================================================================' >> $TESTS_LOG
    printf "%4.4s. %s\n" ${count} ${file} >> $COVERAGE_LOG
    echo '======================================================================' >> $COVERAGE_LOG

    # run coverage
    coverage run -m unittest -v ${file} 2>&1 | tee -a $TESTS_LOG

    echo >> $TESTS_LOG
    echo "\n\n" >> $COVERAGE_LOG

    # report the output
    coverage report -m >> $COVERAGE_LOG

    count=$((count+1))
  done;

  end_time=$(date -u +%s.%M)
  elapsed_time=$(bc <<< "$end_time-$start_time")
  printf "Total time: %10.5f s.\n" $elapsed_time >> $TESTS_LOG
}

function run_all () {
  clean_files
  run_coverage
}

time run_all
