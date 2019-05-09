wget https://www.dropbox.com/s/9qoal0a9exbkw0f/best_bag_0_1.zip -O best_bag_0_1.zip
unzip -o best_bag_0_1
mv best_bag_0_1/word* .
python3 test.py $1 $2 $3
