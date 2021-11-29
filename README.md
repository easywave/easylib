# how to build
Prerequisite
```
sudo apt install llvm-6.0 clang-6.0
```

Compile
```
git clone https://github.com/easywave/easylib.git
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/clang++-6.0 -DCMAKE_C_COMPILER=/usr/bin/clang-6.0 -DCMAKE_INSTALL_PREFIX=`pwd`/install
make install -j 12
```

Run
```
export LD_LIBRARY_PATH=`pwd`/install/easylib/lib:$LD_LIBRARY_PATH
install/easylib/lib/test
```
