for d in /home/yehiahesham/Desktop/vi/ML/Practical_Machine_Deep_Learning/Assignments/Ass3/Data/train/*/ ; do   
    mv $d/images/* $d
    rm -rf $d/images/
    rm $d/*.txt
done