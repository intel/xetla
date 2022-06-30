FROM ubuntu:20.04

# If you are behind a proxy, let apt-get know about it
#ENV http_proxy=http://proxy-us.intel.com:912
#ENV https_proxy=http://proxy-us.intel.com:912

ENV DEBIAN_FRONTEND=noninteractive

# Packages
RUN apt-get -y update
RUN apt-get install --no-install-recommends -y clang-format-10 git

# clean-up caches, don't run apt after this
RUN rm -rf /var/lib/apt/lists/*

RUN echo "" > /xetpp_docker

# Some workaround for current docker instance on gca host root docker
ENV GROUP_ID 1001
ENV USER_ID 1001
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user