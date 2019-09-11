
for i in {4..9}
do
#aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_$i.tar.gz ./
tar xzvf train_$i.tar.gz --directory imgs/
done

for i in a b c d e f
do 
#aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_$i.tar.gz ./
tar xzvf train_$i.tar.gz --directory imgs/
done
