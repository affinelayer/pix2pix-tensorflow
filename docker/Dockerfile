FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

WORKDIR /root

RUN apt-get update

# caffe
# from https://github.com/BVLC/caffe/blob/master/docker/cpu/Dockerfile
RUN apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy

ENV CAFFE_ROOT=/opt/caffe

RUN mkdir -p $CAFFE_ROOT && \
    cd $CAFFE_ROOT && \
    git clone https://github.com/s9xie/hed . && \
    git checkout 9e74dd710773d8d8a469ad905c76f4a7fa08f945 && \
    pip install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    # https://github.com/s9xie/hed/pull/23
    sed -i "s|add_subdirectory(examples)||g" CMakeLists.txt && \
    # https://github.com/s9xie/hed/issues/11
    sed -i "647s|//||" include/caffe/loss_layers.hpp && \
    sed -i "648s|//||" include/caffe/loss_layers.hpp && \
    mkdir build && cd build && \
    cmake -DCPU_ONLY=1 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

RUN cd $CAFFE_ROOT && curl -O http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel

# octave
RUN apt-get install -y --no-install-recommends octave liboctave-dev && \
    octave --eval "pkg install -forge image" && \
    echo "pkg load image;" >> /root/.octaverc

RUN apt-get install -y --no-install-recommends unzip && \
    curl -O https://pdollar.github.io/toolbox/archive/piotr_toolbox.zip && \
    unzip piotr_toolbox.zip && \
    octave --eval "addpath(genpath('/root/toolbox')); savepath;" && \
    echo "#include <stdlib.h>" > wrappers.hpp && \
    cat /root/toolbox/channels/private/wrappers.hpp >> wrappers.hpp && \
    mv wrappers.hpp /root/toolbox/channels/private/wrappers.hpp && \
    mkdir /root/mex && \
    cd /root/toolbox/channels/private && \
    mkoctfile --mex -DMATLAB_MEX_FILE -o /root/mex/convConst.mex convConst.cpp && \
    mkoctfile --mex -DMATLAB_MEX_FILE -o /root/mex/gradientMex.mex gradientMex.cpp && \
    mkoctfile --mex -DMATLAB_MEX_FILE -o /root/mex/imPadMex.mex imPadMex.cpp && \
    mkoctfile --mex -DMATLAB_MEX_FILE -o /root/mex/imResampleMex.mex imResampleMex.cpp && \
    mkoctfile --mex -DMATLAB_MEX_FILE -o /root/mex/rgbConvertMex.mex rgbConvertMex.cpp && \
    octave --eval "addpath('/root/mex'); savepath;"

RUN curl -O https://raw.githubusercontent.com/pdollar/edges/master/private/edgesNmsMex.cpp && \
    octave --eval "mex edgesNmsMex.cpp" && \
    mv edgesNmsMex.mex /root/mex/

# from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
RUN apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip

# gpu tracing in tensorflow
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN pip install \
    appdirs==1.4.0 \
    funcsigs==1.0.2 \
    google-api-python-client==1.6.2 \
    google-auth==0.7.0 \
    google-auth-httplib2==0.0.2 \
    google-cloud-core==0.22.1 \
    google-cloud-storage==0.22.0 \
    googleapis-common-protos==1.5.2 \
    httplib2==0.10.3 \
    mock==2.0.0 \
    numpy==1.12.0 \
    oauth2client==4.0.0 \
    packaging==16.8 \
    pbr==1.10.0 \
    protobuf==3.2.0 \
    pyasn1==0.2.2 \
    pyasn1-modules==0.0.8 \
    pyparsing==2.1.10 \
    rsa==3.4.2 \
    six==1.10.0 \
    uritemplate==3.0.0 \
    tensorflow-gpu==1.0.0

RUN curl -O https://releases.hashicorp.com/terraform/0.8.7/terraform_0.8.7_linux_amd64.zip && \
    unzip terraform_0.8.7_linux_amd64.zip -d /usr/local/bin && \
    rm terraform_0.8.7_linux_amd64.zip