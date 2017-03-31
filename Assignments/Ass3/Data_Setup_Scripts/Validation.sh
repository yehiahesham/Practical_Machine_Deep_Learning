for d in */ ; do
    mkdir /home/yehiahesham/Desktop/vi/ML/Practical_Machine_Deep_Learning/Assignments/Ass3/Data/validate/$d
    for (( i=0; i <= 49; i++ ))   
    do
        mv $d/*_$i.JPEG /home/yehiahesham/Desktop/vi/ML/Practical_Machine_Deep_Learning/Assignments/Ass3/Data/validate/$d
    done
done