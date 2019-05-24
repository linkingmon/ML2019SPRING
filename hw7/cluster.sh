wget -O encoder.h5 'https://github.com/linkingmon/models/releases/download/0.0.2/encoder.h5'
wget -O img.npy 'https://github.com/linkingmon/models/releases/download/0.0.2/img.npy'
python3 cluster.py $1 $2 $3