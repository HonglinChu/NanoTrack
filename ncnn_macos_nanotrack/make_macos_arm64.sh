dirname="build"
if [!-d $dirname];then
    mkdir $dirname
else
    echo dir exist 
fi
cd ./build

cmake  -DCMAKE_SYSTEM_PROCESSOR=arm64 -DCMAKE_OSX_ARCHITECTURES=arm64   .. \

make -j8

cd ../

# DCMAKE_BUILD_TYPE=Release 