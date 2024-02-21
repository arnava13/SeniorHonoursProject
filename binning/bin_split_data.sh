#!/bin/bash
echo "Initializing training data"
echo ""

cd "$(dirname "$0")"

echo $PWD
n=0


mkdir binA
for iteration in {1..5000}
do
echo "Adding ${iteration} to bin A"
n=$((n+1))
cp data_lcdm_ordered/${iteration}.txt binA/${n}.txt
echo "${n}"
done

mkdir binB
for iteration in {1..5000}
do
myi=$((iteration + 5000))
echo "Adding ${myi} to bin B"
n=$((n+1))
cp data_lcdm_ordered/${myi}.txt binB/${iteration}.txt
echo "${n}"
done

mkdir binC
for iteration in {1..5000}
do
myi=$((iteration + 5000 + 5000))
echo "Adding ${myi} to bin C"
n=$((n+1))
cp data_lcdm_ordered/${myi}.txt binC/${iteration}.txt
echo "${n}"
done


mkdir binD
for iteration in {1..5000}
do
myi=$((iteration + 5000 + 5000 + 5000))
echo "Adding ${myi} to bin D"
n=$((n+1))
cp data_lcdm_ordered/${myi}.txt binD/${iteration}.txt
echo "${n}"
done


echo "DONE!!! :D "
