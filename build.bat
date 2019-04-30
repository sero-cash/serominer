@echo off
setlocal

rem add MSVC in PATH
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsMSBuildCmd.bat"

rem add Perl in PATH; it's needed for OpenSSL build
set "PERL_PATH=C:\Perl\perl\bin"
set "CMAKE_PATH=C:\Program Files\CMake\bin"
set "PATH=%PERL_PATH%;%CMAKE_PATH%;%PATH%"

rem switch to ethminer's source folder
rem cd "%~dp0\cminer\"

if not exist "build\" mkdir "build\"

rem For CUDA 9.x pass also `-T v140`
cmake -G "Visual Studio 15 2017 Win64" -H. -Bbuild -T v140 -DCUDA_TOOLKIT_ROOT_DIR="'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0'" ..

cd build

cmake --build . --config RelWithDebInfo --target gminer
rem make cminer

endlocal
pause
