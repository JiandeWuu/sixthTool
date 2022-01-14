# test ConsensusClusterPlus
#####
x1 = c(rnorm(50, 0,0.5), rnorm(10,6,0.5))
x2 = rnorm(60)
class= factor(rep(1:2, each=50))
plot(x1,x2, pch=20, col=class)

df <- data.frame(x1,x2)
d = as.matrix(df)
d = t(d)
dim(d)

results = ConsensusClusterPlus(d,maxK=3, clusterAlg="pam",title="test_run",plot="pngBMP")
M = results[[2]][["consensusClass"]]
plot(d['x1',],d['x2',], pch=20, col=M)



library(M3C)
d = as.matrix(k2p10nor2n10)
d = t(d)
df <- data.frame(d)
test <- M3C(df)

# library
#####
library(ConsensusClusterPlus)
library(parallel)
library(readr)

# data
#####
setwd("R/cluster_output/")
k3p10nor2n10 <- read_csv("~/R/sixthTool/data/linear_features/point/train/k3p10nor2n10.csv")

k1234_PCPseDNCGa_TNCGa_PseDNC <- read_csv("~/R/sixthTool/data/merge_data/k1234_PCPseDNCGa_TNCGa_PseDNC.csv")
d = as.matrix(k3p10nor2n10)
d = t(d)
dim(d)

# ConsensusClusterPlus
#####
seed=11111
maxK = 15

results = ConsensusClusterPlus(d,maxK=maxK,reps=100,title="k3p10nor2n10", clusterAlg="km",seed=seed,plot="pngBMP")

Kvec = 2:maxK
x1 = 0.1; x2 = 0.9 # threshold defining the intermediate sub-interval
PAC = rep(NA,length(Kvec)) 
names(PAC) = paste("K=",Kvec,sep="") # from 2 to maxK
for(i in Kvec){
  M = results[[i]]$consensusMatrix
  Fn = ecdf(M[lower.tri(M)])
  PAC[i-1] = Fn(x2) - Fn(x1)
}#end for i
# The optimal K
optK = Kvec[which.min(PAC)]

M = results[[2]][["consensusClass"]][1:5]
M
class
####
# preparing input data
library(ALL)
data(ALL)
d = exprs(ALL)
d[1:5,1:5]
mads=apply(d,1,mad)
d = d[rev(order(mads))[1:5000],]
d = sweep(d,1, apply(d,1,median,na.rm=T))

# running
library(ConsensusClusterPlus)
title=tempdir()
results = ConsensusClusterPlus(d,maxK=6,reps=50,pItem=0.8,pFeature=1,  title=title,clusterAlg="hc",distance="pearson",seed=1262118388.7127)

#consensusMatrix - the consensus matrix.
#For .example, the top five rows and columns of results for k=2:
results[[2]][["consensusMatrix"]][1:5,1:5]

#consensusClass - the sample classifications
results[[2]][["consensusClass"]][1:5]

# generating cluster and item consensus
icl = calcICL(results,title=title,plot="png")
icl[["clusterConsensus"]][1:5,]


# obtain gene expression data
library(Biobase)
data(geneData)
d = geneData
#median center genes
dc = sweep(d,1, apply(d,1,median))
# run consensus cluster, with standard options
rcc = ConsensusClusterPlus(dc,maxK=4,reps=100,pItem=0.8,pFeature=1,title="example",distance="pearson",clusterAlg="hc", plot="pngBMP")
# same as above but with pre-computed distance matrix, useful for large datasets (>1,000's of items)
dt = as.dist(1-cor(dc,method="pearson"))
rcc2 = ConsensusClusterPlus(dt,maxK=4,reps=100,pItem=0.8,pFeature=1,title="example2",distance="pearson",clusterAlg="hc")
# k-means clustering
rcc3 = ConsensusClusterPlus(d,maxK=4,reps=100,pItem=0.8,pFeature=1,title="example3",distance="pearson",clusterAlg="km")
### partition around medoids clustering with manhattan distance
rcc4 = ConsensusClusterPlus(d,maxK=4,reps=100,pItem=0.8,pFeature=1,title="example3",distance="manhattan",clusterAlg="pam")
## example of custom distance function as hook:
myDistFunc = function(x){ dist(x,method="manhattan")}
rcc5 = ConsensusClusterPlus(d,maxK=4,reps=100,pItem=0.8,pFeature=1,title="example3",distance="myDistFunc",clusterAlg="pam")
##example of clusterAlg as hook: