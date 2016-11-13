#! /bin/sh

dev=$1
id=$2

echo === >> logs/autoencoder.txt
echo python test_autoencoder.py $dev $id >> logs/autoencoder.txt
python test_autoencoder.py $dev $id > tfm/autoencoder.tfm.py
python macros.py autoencoder/Autoencoder > autoencoder/Autoencoder.py 

python autoencoder/AutoencoderRunner.py
echo === >> logs/autoencoder.txt
