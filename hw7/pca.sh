wget -O S.npy 'https://github.com/linkingmon/models/releases/download/0.0.2/S.npy'
wget -O U.npy 'https://github.com/linkingmon/models/releases/download/0.0.2/U.npy'
wget -O V.npy 'https://github.com/linkingmon/models/releases/download/0.0.2/V.npy'
wget -O mean_face.npy 'https://github.com/linkingmon/models/releases/download/0.0.2/mean_face.npy'
python3 pca.py $1 $2 $3