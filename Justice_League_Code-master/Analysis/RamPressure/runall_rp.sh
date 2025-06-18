#!/bin/bash

cd ~/Research/Justice_League_Code/Analysis/RamPressure/
pwd
date

python particletracking_stars.py h242 41 & #1 
python particletracking_stars.py h329 33 & #2
python particletracking_stars.py h229 27 & #3
python particletracking_stars.py h148 13 & #4
wait
python particletracking_stars.py h229 20 & #5
python particletracking_stars.py h242 24 & #6
python particletracking_stars.py h229 22 & #7
wait
python particletracking_stars.py h242 80 & #8
python particletracking_stars.py h148 37 & #9
python particletracking_stars.py h148 28 & #10 
python particletracking_stars.py h148 68 & #11
wait
python particletracking_stars.py h148 278 & #12
python particletracking_stars.py h229 23 & #13
python particletracking_stars.py h148 45 & #14
wait
python particletracking_stars.py h148 283 & #15
python particletracking_stars.py h229 55 & #16
python particletracking_stars.py h329 137 & #17
wait
python particletracking_stars.py h148 80 & #18
python particletracking_stars.py h148 329 & #19
wait

# include `wait` in between commands to do them in batches
# by my estimate each command uses at most 5-7% of the memory on quirm, so don't run more than 10 at once (ideally less than 10)