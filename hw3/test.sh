rm -r model.zip
wget https://www.dropbox.com/s/hrdgu9dyf6drnx8/model.zip
unzip model
python3 test.py $1 $2
