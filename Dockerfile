FROM darshanpv/rasa-nlu:1.0 as base

RUN mkdir -p /root/server
WORKDIR /root/server

EXPOSE 5001
