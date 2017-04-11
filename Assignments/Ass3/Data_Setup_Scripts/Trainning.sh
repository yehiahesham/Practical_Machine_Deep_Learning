for d in /home/yehiahesham/Desktop/ML/Ass3_data/train/*/ ; do
    mv $d/images/* $d
    rm -rf $d/images/
    rm $d/*.txt
done
