# Usage: Rscript castle_cell_type_annotation.r organ

# parse ordered arguments
args <- commandArgs(trailingOnly=TRUE)
organ <- args[1]

suppressPackageStartupMessages(library(scater))
suppressPackageStartupMessages(library(xgboost))
suppressPackageStartupMessages(library(igraph))
BREAKS=c(-1, 0, 1, 6, Inf)
nFeatures = 100

print(paste("Training ", organ, sep=""))

# import training and test data
rootdir="/path/to/data/"
train_counts <- t(as.matrix(read.csv(file = paste(rootdir, organ, "_filtered_data_train.csv", sep=""), row.names = 1)))
test_counts <- t(as.matrix(read.csv(file = paste(rootdir, organ, "_filtered_data_test.csv", sep=""), row.names = 1)))
train_celltype <- as.matrix(read.csv(file = paste(rootdir, organ, "_filtered_celltype_train.csv", sep="")))
test_celltype <- as.matrix(read.csv(file = paste(rootdir, organ, "_filtered_celltype_test.csv", sep="")))

# select features
sourceCellTypes = as.factor(train_celltype[,"Cell_type"])
ds = rbind(train_counts,test_counts)
ds[is.na(ds)] <- 0
isSource = c(rep(TRUE,nrow(train_counts)), rep(FALSE,nrow(test_counts)))
topFeaturesAvg = colnames(ds[isSource,])[order(apply(ds[isSource,], 2, mean), decreasing = T)]
topFeaturesMi = names(sort(apply(ds[isSource,],2,function(x) { compare(cut(x,breaks=BREAKS),sourceCellTypes,method = "nmi") }), decreasing = T))
selectedFeatures = union(head(topFeaturesAvg, nFeatures) , head(topFeaturesMi, nFeatures) )
tmp = cor(ds[isSource,selectedFeatures], method = "pearson")
tmp[!lower.tri(tmp)] = 0
selectedFeatures = selectedFeatures[apply(tmp,2,function(x) any(x < 0.9))]
remove(tmp)

# bin expression values and expand features by bins
dsBins = apply(ds[, selectedFeatures], 2, cut, breaks= BREAKS)
nUniq = apply(dsBins, 2, function(x) { length(unique(x)) })
ds = model.matrix(~ . , as.data.frame(dsBins[,nUniq>1]))
remove(dsBins, nUniq)

# train model
train = runif(nrow(ds[isSource,]))<0.8
# slightly different setup for multiclass and binary classification
if (length(unique(sourceCellTypes)) > 2) {
  xg=xgboost(data=ds[isSource,][train, ] ,
       label=as.numeric(sourceCellTypes[train])-1,
       objective="multi:softmax", num_class=length(unique(sourceCellTypes)),
       eta=0.7 , nthread=5, nround=20, verbose=0,
       gamma=0.001, max_depth=5, min_child_weight=10)
} else {
  xg=xgboost(data=ds[isSource,][train, ] ,
       label=as.numeric(sourceCellTypes[train])-1,
       eta=0.7 , nthread=5, nround=20, verbose=0,
       gamma=0.001, max_depth=5, min_child_weight=10)
}

# validate model
predictedClasses = predict(xg, ds[!isSource, ])
testCellTypes = as.factor(test_celltype[,"Cell_type"])
trueClasses <- as.numeric(testCellTypes)-1

cm <- as.matrix(table(Actual = trueClasses, Predicted = predictedClasses))
n <- sum(cm)
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
accuracy = sum(diag) / n
precision = diag / colsums
recall = diag / rowsums
f1 = 2 * precision * recall / (precision + recall)
macroF1 = mean(f1)

print(paste(organ, " accuracy: ", accuracy, sep=""))
print(paste(organ, " macroF1: ", macroF1, sep=""))

results_df = data.frame(Accuracy=c(accuracy),macroF1=c(macroF1))
write.csv(results_df,paste(rootdir, organ, "_castle_results_test.csv", sep=""), row.names = FALSE)
