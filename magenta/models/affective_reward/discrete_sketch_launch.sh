#!/bin/bash

#declare -a arr=("cat" "bird" "bicycle" "octopus" "face" "flamingo" "cruise_ship" "truck" "pineapple" "spider" "mosquito" "angel" "butterfly" "pig" "garden" "The_Mona_Lisa" "crab" "windmill" "yoga" "hedgehog" "castle" "ant" "basket" "chair" "bridge" "diving_board" "firetruck" "flower" "owl" "palm_tree" "pig" "rain" "skull" "duck" "snowflake" "speedboat" "sheep" "scorpion" "sea_turtle" "pool" "paintbrush" "bee" "backpack" "ambulance" "barn" "bus" "cactus" "calendar" "couch" "hand" "helicopter" "lighthouse" "lion" "parrot" "passport" "peas" "postcard" "power_outlet" "radio" "snail" "stove" "strawberry" "swan" "swing_set" "tiger" "toothpaste" "toothbrush" "trombone" "whale" "tractor" "squirrel" "alarm_clock" "bear" "book" "brain" "bulldozer" "dog" "dolphin" "elephant" "eye" "fan" "fire_hydrant" "frog" "kangaroo" "key" "lantern" "lobster" "map" "mermaid" "monkey" "penguin" "rabbit" "rhinoceros" "rifle" "roller_coaster" "sandwich" "steak")
#declare -a arr=("cat" "penguin" "rhinoceros" "crab" "face" "horse" "owl")
declare -a arr=("face" "duck" "cat" "owl" "crab" "frog" "rabbit" "penguin")

for name in "${arr[@]}"
  do
    echo "Starting a job for model $name"
    discrete_sketch_train.py #TODO: args here about "data_class=$name"
  done

