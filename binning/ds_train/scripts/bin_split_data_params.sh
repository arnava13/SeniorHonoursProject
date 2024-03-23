#!/bin/bash
echo "Initializing training data"
echo ""

cd "$(dirname "$0")"

echo $PWD
n=0
indir=ds_train_ordered_4k
n_perbin=1000
cosmologyFile=cosmo_ds_train_ordered.txt

mkdir binA
binAFile="binA.txt"
for iteration in $(seq 1 $n_perbin)
do
echo "Adding ${iteration} to bin A"
n=$((n+1))
cp $indir/${iteration}.txt binA/${n}.txt
echo "${n} $(awk -v id="${iteration}" 'NR==id {print $0}' $cosmologyFile)" >> $binAFile
done

mkdir binB
binBFile="binB.txt"
for iteration in $(seq 1 $n_perbin)
do
myi=$((iteration + $n_perbin))
echo "Adding ${myi} to bin B"
cp $indir/${myi}.txt binB/${iteration}.txt
echo "${iteration} $(awk -v id="${myi}" 'NR==id {print $0}' $cosmologyFile)" >> $binBFile
done

mkdir binC
binCFile="binC.txt"
for iteration in $(seq 1 $n_perbin)
do
myi=$((iteration + 2*$n_perbin))
echo "Adding ${myi} to bin C"
cp $indir/${myi}.txt binC/${iteration}.txt
echo "${iteration} $(awk -v id="${myi}" 'NR==id {print $0}' $cosmologyFile)" >> $binCFile
done

mkdir binD
binDFile="binD.txt"
for iteration in $(seq 1 $n_perbin)
do
myi=$((iteration + 3*$n_perbin))
echo "Adding ${myi} to bin D"
cp $indir/${myi}.txt binD/${iteration}.txt
echo "${iteration} $(awk -v id="${myi}" 'NR==id {print $0}' $cosmologyFile)" >> $binDFile
done

echo "DONE!!! :D "
