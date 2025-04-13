#!/bin/bash
cur_dir=$(dirname -- $0)
cd $(dirname -- $0)

echo running "python unit test scripts" in $PWD
UnittestPyfiles=(test_*.py)
boarder=======================================================================
echo "--- Unit test python scripts"
for pyfile in "${UnittestPyfiles[@]}"
do
    echo python $cur_dir/$pyfile
done
echo "---"

# read -p "Press [Enter] key to continue."

for pyfile in "${UnittestPyfiles[@]}"
do
    echo $boarder
    echo ======= Processing $pyfile
    python3 $pyfile
    if [ $? -eq 0 ]
       then
           echo $boarder
    else
        echo ======= Check $(readlink -f $pyfile)
        echo ''
        exit
    fi
    echo ''
done
