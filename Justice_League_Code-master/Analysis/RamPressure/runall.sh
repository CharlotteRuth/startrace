#!/bin/bash

cd ~/Research/Justice_League_Code/Analysis/RamPressure/
pwd
date

# updated to use new haloids Dec 31 2022
# python particletracking.py h242 38 & # 1
# python particletracking.py h329 29 & # 2
# python particletracking.py h229 22 & # 3
# python particletracking.py h148 12 & # 4
# wait
# python particletracking.py h229 14 & # 5
# python particletracking.py h242 21 & # 6
# python particletracking.py h229 18 & # 7
# python particletracking.py h242 69 & # 8
# wait
# python particletracking.py h148 34 & # 9
# python particletracking.py h148 27 & # 10
# python particletracking.py h148 55 & # 11
# python particletracking.py h148 251 & # 12
# wait
# python particletracking.py h229 20 & # 13
# python particletracking.py h148 38 & # 14
# python particletracking.py h148 249 & # 15
# python particletracking.py h229 49 & # 16
# wait
# python particletracking.py h329 117 & # 17
# python particletracking.py h148 65 & # 18
# python particletracking.py h148 282 & # 19
# wait

python particletracking.py h148 2 &
python particletracking.py h242 30 &
python particletracking.py h148 6 &
python particletracking.py h329 7 &
wait 
python particletracking.py h148 3 &
python particletracking.py h148 7 &
python particletracking.py h242 401 &
python particletracking.py h242 8 &
wait 
python particletracking.py h148 4 &
python particletracking.py h148 10 &
python particletracking.py h242 10 &
wait


# include `wait` in between commands to do them in batches
# by my estimate each command uses at most 5-7% of the memory on quirm, so don't run more than 10 at once (ideally less than 10)
