for d in /home/yehiahesham/Desktop/vi/ML/Ass3_data/train/* ; do
    for (( i=0; i <= 49; i++ ))
    do
        mv $d/*_$i.JPEG /home/yehiahesham/Desktop/vi/ML/Ass3_data/validate/
    done
done