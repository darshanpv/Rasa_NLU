FROM darshanpv/rasa-nlu:1.0 as base

ADD server/ /root/server/
WORKDIR /root/server

EXPOSE 5001

ENTRYPOINT ["python"]
CMD ["app.py"]