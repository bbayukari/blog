本文是用来指导安装https://github.com/kthohr/optim 的说明文档，主要给新手或者三个月后的自己看，所以这里说活风格可能随意一些，不太严谨。

首先，如果你是常在linux命令行间驰骋，各种库各种语言开发的大佬，那就没必要看本文了，直接参考项目里的安装说明即可。
其次，如果你是从没用过命令行，没接触过Linux，可能宇宙第一IDE---Visual Studio的教程更适合你，不过我没弄懂，要是成功编译出了dll文件的话请教教我。

不过从这段代码来看vs可能不是个好选择。
```
#ifdef _MSC_VER
    #error OptimLib: MSVC is not supported
#endif
```

现在正式开始，先来看一下关键的问题：核心难点就在于项目作者只提供了Linux下的安装方式，所以如果你手上有Linux系统的PC，服务器，就好说了，Mac的话，没用过，不过听说跟Linux比较像，估计也行。所以如果你只有windows系统的电脑，或者对Linux不熟悉虽然有但没怎么用过，你可以接着看了。

经过一整天的调查学习，我得到了三种安装方式，并终于在凌晨成功了其中一种方式，其他两种方式有时间再更。先说说那两种失败的方法吧。

## CMAKE

首先上介绍：CMake is an open-source, cross-platform family of tools designed to build, test and package software. CMake is used to control the software compilation process using simple platform and compiler independent configuration files, and generate native makefiles and workspaces that can be used in the compiler environment of your choice. The suite of CMake tools were created by Kitware in response to the need for a powerful, cross-platform build environment for open-source projects such as ITK and VTK.

关键词cross-platform跨平台，这条路应该能成，因为那个项目有个issue，作者给了个CMakeList.txt，位置是https://github.com/kthohr/optim/issues/11。

## MSYS2

还是上介绍：MSYS2 is a collection of tools and libraries providing you with an easy-to-use environment for building, installing and running native Windows software. It provides a native build environment, based on open source software, and makes you feel right at home when you are already comfortable with Linux. 

我的理解就是虽然我在windows里，但操作就像Linux一样。下载的话，可以直接去USTC镜像 https://mirrors.ustc.edu.cn/help/msys2.html 。使用教程可以看看这个，http://www.360doc.com/content/16/0514/16/496343_559090195.shtml， 其实写的挺一般的，建议还是去其他地方多搜搜。

msys2的基本环境设置好后，就和Linux差不多了，就放到下面一块说了。另外，这条路我已经快走通了，只是它要我下载两个基础线性代数的包，blas和lapack，相关包有好多，我下了好几个都不对，实在过不去了。

## Linux

说来说去，还是在Linux上搞容易，建议通过
1. WSL: window subsystem of linux
2. Linux 虚拟机
3. Linux 服务器
方式拿到一个Linux环境，教程很多，这里就不说了。

首先是安装工具链，就是git,gcc,g++,make之类的，可以通过sudo apt-get install + “name"的方式安装，基本上用到时缺啥安啥就行了，有些可能你的Linux发行版自带的有就不用装了。举个例子，到后边编译的时候，会返回错误信息cannot find -lblas， cannot find -llapack啥的，就是说缺少库blas,lapack,遇到之后，运行sudo apt-get install libblas-dev liblapack-dev就行了。

然后是安装依赖，如果要用矩阵运算，需要Eigen或者Armadillo。如果直接用apt安装Eigen，版本号可能达不到3.4.0，巨坑。所以，这里就用最复杂的从源码安装的方式举例。Eigen库只需要包括头文件，需要做的就只有：从官网下载最新的源码，复制到Linux系统里，把其中的头文件部分--一个叫Eigen的文件夹（里面有个叫Dense的文件）放到默认路径/usr/include里。

windows和linux交换文件有很多方法，跟你得到Linux系统的方式有关。如果你和我一样用VirtualBox，可以用共享文件夹来实现。这里有个坑，想用共享文件夹就要安装VB的增强功能，本来是图形化界面的事，在我这儿不知道为啥失败了，找了好久的教程，终于成了。步骤是：1. 找到VirtualBox的增强文件 VBoxGuestAdditions.iso，这是一个光盘文件，然后让虚拟机读取他，相当于给虚拟机插了一张光盘。我们要运行这个光盘里的文件VBoxLinuxAdditions.run， 必须先挂载！具体命令为：
```
sudo mkdir --p /media/cdrom
sudo mount -t auto /dev/cdrom /media/cdrom/
cd /media/cdrom/
sudo sh VBoxLinuxAdditions.run
```
这样，VB增强功能就安装好了，我们可以使用它的共享文件夹功能，在VB里设置好后，在虚拟机里运行sudo mount -t  vboxsf  share /home/yukari/share即可。

终于，可以开始按照项目的说明来了：进入工作目录，然后运行
```
git clone https://github.com/kthohr/optim ./optim
cd ./optim
git submodule update --init
export EIGEN_INCLUDE_PATH=/usr/include
./configure -i "/usr/local" -l eigen -p
make
make install
```
这样我们应该得到了一个/usr/local/lib/liboptim.so文件，此外记下optim头文件的绝对位置，他应该在./include，把这个目录记作/path/to/include。

安装完成后，可以开始编写cpp程序了，把它存在main.cpp里，接着执行
>g++ -march=native -Ipath/to/include main.cpp -o test -L/usr/local/lib -loptim

得到可执行文件test，接着

>./test > result.txt

这样就把结果存在result.txt里了。
