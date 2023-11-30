#!/bin/bash
total=$(ls $1 -1 | wc -l)
echo $total
counter=1
for FILE in $1*; do
	outname="$2$(basename ${FILE})"
	ontogpt extract -i $FILE -t desiccation > $outname
	echo $counter
	$((counter++))
done | tqdm --total $total
>> /dev/null

