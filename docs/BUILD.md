# Building from source

## Table of Contents

* [Requirements](#requirements)
    * [Common](#common)
    * [Linux](#linux)
        * [OpenCL support on Linux](#opencl-support-on-linux)
    * [macOS](#macos)
    * [Windows](#windows)
* [CMake configuration options](#cmake-configuration-options)
* [Disable Hunter](#disable-hunter)
* [Instructions](#instructions)
    * [Windows-specific script](#windows-specific-script)


## Requirements

This project uses [CMake] and [Hunter] package manager.

### Common

1. [CMake] >= 3.5
2. [Git](https://git-scm.com/downloads)
3. [Perl](https://www.perl.org/get.html), needed to build OpenSSL
4. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >= 9.0 (optional, install if you want NVidia CUDA support)

### Linux

1. GCC version >= 4.8
2. DBUS development libs if building with `-DDBUS`. E.g. on Ubuntu run:

```shell
sudo apt install libdbus-1-dev
```

#### OpenCL support on Linux

If you're planning to use [OpenCL on Linux](https://github.com/ruslo/hunter/wiki/pkg.opencl#pitfalls)
you have to install the OpenGL libraries. E.g. on Ubuntu run:

```shell
sudo apt-get install mesa-common-dev
```

### macOS

1. GCC version >= TBF

### Windows

1. [Visual Studio 2017](https://www.visualstudio.com/downloads/); Community Edition works fine. **Make sure you install MSVC 2015 toolkit (v140).**

## Instructions

1. Make sure git submodules are up to date:

    ```shell
    git submodule update --init --recursive
    ```

2. Create a build directory:

    ```shell
    mkdir build
    cd build
    ```

3. Configure the project with CMake. Check out the additional [configuration options](#cmake-configuration-options).

    ```shell
    cmake ..
    ```

    **Note:** On Windows, it's possible to have issues with VS 2017 default compilers, due to CUDA expecting a specific toolset version; in that case, use the VS 2017 installer to get the VS 2015 compilers and pass the `-T v140` option:

    ```shell
    cmake .. -G "Visual Studio 15 2017 Win64"
    # or this if you have build errors in the CUDA step
    cmake .. -G "Visual Studio 15 2017 Win64" -T v140
    ```

4. Build the project using [CMake Build Tool Mode]. This is a portable variant of `make`.

    ```shell
    make serominer
    ```

    Note: On Windows, it is possible to have compiler issues if you don't specify the build config. In that case use:

    ```shell
    cmake --build . --config Release --target serominer
    ```


### Windows-specific script

Complete sample Windows batch file - **adapt it to your system**. Assumes that:

* it's placed one folder up from the serominer source folder
* you have CMake installed
* you have Perl installed

See "build-cuda.bat" or "build-opencl.bat"

## CMake configuration options

Pass these options to CMake configuration command, e.g.

```shell
cmake .. -DCUDA=ON -DOPENCL=OFF
```

* `-DOPENCL=ON` - enable OpenCL mining, `ON` by default.
* `-DCUDA=ON` - enable CUDA mining, `ON` by default.
* `-DAPI=ON` - enable API Server, `ON` by default.
* `-DBINKERN=ON` - install AMD binary kernels, `ON` by default.
* `-DDBUS=ON` - enable D-Bus support, `OFF` by default.

## Disable Hunter

If you want to install dependencies yourself or use system package manager you can disable Hunter by adding
[`-DHUNTER_ENABLED=OFF`](https://docs.hunter.sh/en/latest/reference/user-variables.html#hunter-enabled)
to the configuration options.


[CMake]: https://cmake.org/
[CMake Build Tool Mode]: https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-tool-mode
[Hunter]: https://docs.hunter.sh/
