# DeepLearning
DeepLearning

conda update --all --yes

conda config --set channel_priority false

Export conda packages without build
conda env export --no-builds > environment.yml

Conda create env
conda env create -f environment.yml

docker build . --tag conda-brain:0.0.1

docker run -it conda-brain:0.0.1 bash

conda create -n <environment-name> --file req.txt

 docker cp /hostfile  (container_id):/(to_the_place_you_want_the_file_to_be)