#!/bin/bash

mkdir build
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release
cd build
make -j17 exe
cp ./Release/bin/exe ..
cd ..

if [ "$1" == "clean" ]
then
   rm -r ./build
fi
