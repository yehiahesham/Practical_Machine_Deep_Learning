mkdir /home/yehiahesham/Desktop/ML/Ass3_data/validate/
cd /home/yehiahesham/Desktop/ML/Ass3_data/validate/
for d in /home/yehiahesham/Desktop/ML/Ass3_data/train/* ; do
	mkdir ${d#/home/yehiahesham/Desktop/ML/Ass3_data/train/}
    for (( i=0; i <= 49; i++ ))
    do
        mv $d/*_$i.JPEG ${d#/home/yehiahesham/Desktop/ML/Ass3_data/train/}
    done
done
