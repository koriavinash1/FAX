jobs=$(squeue | grep agk21)

re='^[0-9]+$'

for job in $jobs; do 
	if [[ $job =~ $re ]]; then
		echo $job;
		scancel $job
	fi

done
