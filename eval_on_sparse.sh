#!/usr/bin/env bash
# declare an array variable
declare -a envs=(
    "maze2d-open-v0"
    "maze2d-umaze-v1"
    "maze2d-medium-v1"
    "maze2d-large-v1"
    "maze2d-open-dense-v0"
    "maze2d-umaze-dense-v1"
    "maze2d-medium-dense-v1"
    "maze2d-large-dense-v1"
	)
 
# get length of an array
length=${#envs[@]}
 
# use C style for loop syntax to read all values and indexes
for (( j=0; j<length; j++ ));
do
  if [ $j -lt 4 ];  then 
    printf "Sparse"
    python main.py \
    --env ${envs[$j]} \
    --eval_env ${envs[$j]} \
    #--save_model True
    #printf "Current index %d with value %s\n" $j "${envs[$j+4]}"
  else
    printf "Dense"
    python main.py \
    --env ${envs[$j]} \
    --eval_env ${envs[$j-4]} \
    #--save_model True
    #printf "Current index %d with value %s\n" $j "${envs[$j]}"
  fi      
done