FROM quay.io/spivegin/gitonly:latest AS git

FROM dart:3.8.1
WORKDIR /opt/gcx
RUN ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && git config --global user.name "quadtone" && git config --global user.email "quadtone@txtsme.com"
COPY --from=git /root/.ssh /root/.ssh
#ADD https://github.com/caddyserver/caddy/releases/download/v2.5.1/caddy_2.5.1_linux_amd64.tar.gz /opt/gcx/
ENV deploy=c1f18aefcb3d1074d5166520dbf4ac8d2e85bf41 

RUN ssh-keyscan -H github.com > ~/.ssh/known_hosts &&\
    ssh-keyscan -H gitlab.com >> ~/.ssh/known_hosts &&\
    ssh-keyscan -H gitea.com >> ~/.ssh/know_hosts &&\
    ssh-keyscan -H api.smsburr.com >> ~/.ssh/know_hosts

RUN git config --global url.git@github.com:.insteadOf https://github.com/ &&\
    git config --global url.git@gitlab.com:.insteadOf https://gitlab.com/ &&\
    git config --global url.git@gitea.com:.insteadOf https://gitea.com/ &&\
    git config --global url."https://${deploy}@sc.tpnfc.us/".insteadOf "https://sc.tpnfc.us/"

RUN mkdir -p /opt/tmp /opt/src /opt/go/bin
ENV GOPATH=/opt/src/ \
    GOBIN=/opt/go/bin \
    PATH=/opt/go/bin:$PATH \
    GO_VERSION=1.24.4 \
    GOPROXY=direct \
    GOSUMDB=off

ADD https://storage.googleapis.com/golang/go${GO_VERSION}.linux-amd64.tar.gz /opt/tmp/

# RUN wget -qO- https://dl-ssl.google.com/linux/linux_signing_key.pub \
#   | sudo gpg  --dearmor -o /usr/share/keyrings/dart.gpg &&\
#   echo 'deb [signed-by=/usr/share/keyrings/dart.gpg arch=amd64] https://storage.googleapis.com/download.dartlang.org/linux/debian stable main' \
#   | sudo tee /etc/apt/sources.list.d/dart_stable.list


RUN apt update -y && apt-get -y upgrade  && apt-get install -y unzip curl git && apt upgrade -y &&\
    tar -C /opt/ -xzf /opt/tmp/go${GO_VERSION}.linux-amd64.tar.gz &&\
    chmod +x /opt/go/bin/* &&\
    ln -s /opt/go/bin/* /bin/ &&\
    rm /opt/tmp/go${GO_VERSION}.linux-amd64.tar.gz &&\
    apt-get autoclean && apt-get autoremove &&\
    rm -rf /tmp/* /var/lib/apt/lists/* /var/tmp/*