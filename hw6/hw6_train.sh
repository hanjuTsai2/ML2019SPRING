# bash hw6_train.sh <train_x file> <train_y file> <test_x.csv file> <dict.txt.big file>
python3 train-torch.py $1 $2 $3 $4 --num_layers 4 --hidden_dim 150 --seq_len 50 #--word_dim 1200 #best