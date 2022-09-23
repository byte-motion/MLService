# ML Service
Pytorch inferenced exposed over grpc

## Image notes
Docker image should match the CUDAtoolkit version running on the pc

[See this page for pytorch-to-CUDAtoolkit matching](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

[Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)

## Windows 10 Installation
- Follow [these instructions](https://docs.nvidia.com/CUDA/wsl-user-guide/index.html)
- If the following error is encountered during [this step](https://docs.nvidia.com/CUDA/wsl-user-guide/index.html#running-containers), exec this command:

```
sudo mkdir /sys/fs/cgroup/systemd
sudo mount -t cgroup -o none,name=systemd cgroup /sys/fs/cgroup/systemd
```
- Installation is successful when step [6.1](https://docs.nvidia.com/CUDA/wsl-user-guide/index.html#running-simple-containers) is successful!
- Make sure to disable auto updates on windows drivers:
    Control Panel > System and Security > System > Advanced System Settings > Hardware > Select "No (your device might not work as expected)"

- Make sure to select the Beta Windows preview ring after latest Dev ring was installed as per the installation guide above.

- Enabled live restore in `/etc/docker/daemon.json`:
```
{
  "live-restore": true
}
```

## Linux Installation
- Install [nvidia docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- Install [CUDA-drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)


### Docker Run
MUST have CUDA toolkit version compatible with the [nvcr.io/nvidia/pytorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) docker image container version, see [this table](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) for current container version toolkit compatabilty
To run the latest build from the [bytemotion docker hub repository](https://registry.hub.docker.com/repository/docker/bytemotion/ocellus_ml_service)
This will work in WSL 2 with CUDA enabled as well as in any linux docker
- Windows
```
sudo docker run --restart=unless-stopped --name ocellus_ml_service --gpus all -p 0.0.0.0:50055:50055 -v /mnt/c/Users/<your-user-name>/AppData/LocalLow/Byte\ Motion/Ocellus:/mnt/ocellus -it bytemotion/ocellus_ml_service:<tag>
```
- Linux
```
docker run -d --memory="5g" --memory-swap="5g" --restart=unless-stopped --name ocellus_ml_service --gpus all -p 0.0.0.0:50055:50055 -v ~/.config/unity3d/Byte\ Motion/Ocellus:/mnt/ocellus -it bytemotion/ocellus_ml_service:latest
```

This will make the service run each time docker starts or restarts automatically

### Windows WSL 2 Auto Run
To automatically run docker on windows start:
- Press the Windows logo key + R, type shell:startup
- Create a file in the opened directory called `docker.bat` and paste the following:

```
wsl echo <sudo-password> ^| sudo -S service docker start
```
- `<sudo-password>` should be replaced with the linux user password chosen when installing WSL 2


### Docker Build in WSL Ubuntu
Start a WSL2 terminal and navigate to the ./MLService directory and execute:
```
sudo docker build -t bytemotion/ocellus_ml_service:latest -t bytemotion/ocellus_ml_service:<version> .
```
To push this to docker hub:
```
sudo docker login --username bytemotion
sudo docker push bytemotion/ocellus_ml_service
```
- Note that `bytemotion/ocellus_ml_service` should be incremented with a version number like so: `bytemotion/ocellus_ml_service:v0` where `v0` is incremented, `v1`, `v2` etc

## Development

### Basic Dependencies
- Install CUDA (e.g. 11.2)
- Install supporting [https://developer.nvidia.com/CUDA-toolkit-archive](CUDA toolkit)
- Install [https://developer.nvidia.com/cudnn](cuDNN) (requires nvidia account). [https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux](instructions)
- Download LibTorch from [the pytorch website](https://pytorch.org/get-started/locally/) matching your local CUDA version, extract, and place the entire libtorch in project root (gitignored).
- NOTE: Models should be built using same version or torch as the one you are using for development
- Init submodules: `git submodule init submodules/grpc` and `git submodule update submodules/grpc`
- Checkout `v1.29.1` tag in grpc
- Navigate to `submodules/grpc` and run `git submodule update`
- Intel [MKL Library](https://software.seek.intel.com/performance-libraries)

### Windows
- Install Visual Studio 2019

### CMake
- Download and install `CMake` from [cmake.org/download](https://cmake.org/download/)
- Run cmake-gui from an anaconda terminal
- Set Tourch_DIR to `<libtorch-folder*>\libtorch\share\cmake\Torch`
- Add GRPC_FETCHCONTENT and set to `true`
- Add MKL_INCLUDE_DIR and set to `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl\include`
- *libtorch-folder: location where archive extracted to in basic dependencies instructions
- Select generator `Visual Studio 16 2019`

### Visual Studio
- Open generated project and build a _Release_ version of Caffe2Wrapper (Debug does not work)
- Copy .dll files generated by the release build to into `Assets\Plugins\Caffe2Wrapper`, replacing all old files
