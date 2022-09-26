## cd-hit

https://github.com/weizhongli/cdhit/wiki/3.-User's-Guide

sudo docker run -d -p 8787:8787 -v /data/jand:/home/rstudio/R -e PASSWORD=zxczxc -e ROOT=TRUE -e USERID=1000 --name rstudio-jand rocker/rstudio


kill $(ps -A -ostat,ppid | awk '/[zZ]/ && !a[$2]++ {print $2}')
kill $(ps aux| grep "python hpo_svm_train.py" | awk '{print $2}')
["Cytosolic", "Nucleus"], [0, 1]