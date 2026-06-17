#! /bin/tcsh 

foreach n ( 0 1 2 3 4 )

cat c00$n > test

foreach m ( `echo $n | gawk '{for ( i=0; i<5; i++ ) { if ( i != $1) { print i}}}' ` )

cat c00$m > eval

touch train
rm -f train

foreach l ( `echo $n $m | gawk '{ for ( i=0; i<5; i++ ) { if ( i != $1 && i != $2) { print i}}}' ` )

cat c00$l >> train

end

##exit

#run train-code
#run test evaluation

end

# Pick model with best test perf

# Run evalaution evaluation

end
