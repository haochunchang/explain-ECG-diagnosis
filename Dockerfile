FROM pytorchlightning/pytorch_lightning:latest-py3.8-torch1.5

WORKDIR /usr/src/explain-ecg-diagnosis

COPY ./ /usr/src/explain-ecg-diagnosis
RUN pip install -r requirements.txt

CMD ["/bin/bash"]