wget -O model.h5 'https://github.com/linkingmon/models/releases/download/0.0.1/model.h5'
wget -O model1.h5 'https://github.com/linkingmon/models/releases/download/0.0.1/model1.h5'
wget -O model4.h5 'https://github.com/linkingmon/models/releases/download/0.0.1/model4.h5'
wget -O model5.h5 'https://github.com/linkingmon/models/releases/download/0.0.1/model5.h5'
wget -O model8.h5 'https://github.com/linkingmon/models/releases/download/0.0.1/model8.h5'
python3 ens_test.py $1 $2 $3 $4