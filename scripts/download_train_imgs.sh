
for i in {3..9}
do
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_$i.tar.gz ./
done

for i in a b c d e f
do 
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_$i.tar.gz ./
done
