FROM ubuntu:22.04

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y build-essential git python3 python3-pip wget
RUN pip3 install torch torchvision torchaudio lief numpy tqdm

# RUN mkdir dataset && cd dataset && \
#     wget --no-verbose --no-parent --recursive --reject html,signature --no-check-certificate https://security.ece.cmu.edu/byteweight/elf_64/

ENTRYPOINT ["/bin/bash"]