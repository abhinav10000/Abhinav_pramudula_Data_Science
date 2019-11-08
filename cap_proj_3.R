train <- read.csv('C:\\Users\\alekhyamini\\Downloads\\Credit_Bureau_Data_Train.csv')
val <- read.csv('C:\\Users\\alekhyamini\\Downloads\\Credit_Bureau_Data_Validation.csv')

dim(train)
dim(val)

train = train[,-c(2,3,4,5,7,8,9)]
val = val[,-c(2,3,4,5,7,8,9)]

str(train)




#=======
#SMOTE
#=======

library(DMwR)
?SMOTE
train$Defaulter = as.factor(train$Defaulter)
smote_cr = SMOTE(Defaulter~., data = train, perc.over = 450,perc.under = 150)
table(smote_cr$Defaulter)
table(train$Defaulter)
head(smote_cr)

#=======================================================================================================================================
#Tomek balancing on the imbalanced train data and trying different models to check for improvement in the model performance
#=======================================================================================================================================
#train = train[-3]
names(train)
library(unbalanced)
tomek_bal = ubTomek(X = train[,-1], Y = train[,1])

head(train)
?ubTomek

newData<-cbind(tomek_bal$X, tomek_bal$Y)
View(newData)
table(train$Defaulter)
table(newData$`tomek_bal$Y`)

tomek_cr = newData
head(tomek_cr)
dim(tomek_cr)
colnames(tomek_cr)[18] <- "Defaulter"  
names(tomek_cr)
table(tomek_cr$Defaulter)

#===============================================================================================================
#ENN #Edited Nearest Neighbour
#======================================
en = ubENN(X = train[,-1], Y = train[,1],k = 2)
head(en)
enn_cr = cbind(en$X, en$Y)
?ubENN
table(train$Defaulter)
table(en$Y)
colnames(enn_cr)[18] <- "Defaulter" 
?ubENN
table(enn_cr$Defaulter)


#==================================
#SMOTE + TOMEK
#==================================
head(smote_cr)
smotetomek = ubBalance(X = smote_cr[,-1], Y = smote_cr[,1], type = 'ubTomek')
table(smotetomek$Y)
smotetom_cr = cbind(smotetomek$X,smotetomek$Y)
colnames(smotetom_cr)[18] <- "Defaulter" 
head(smotetom_cr)

#=============================
#SMOTE + ENN
#=============================
smoteenn = ubBalance(X = smote_cr[,-1], Y = smote_cr[,1], type = 'ubENN')
table(smoteenn$Y)
smoteenn_cr = cbind(smoteenn$X,smoteenn$Y)
colnames(smoteenn_cr)[18] <- "Defaulter" 


######################### LOGISTIC REGRESSION ####################################

eval_mat = matrix()
#Building logisitic regression model on the  data  
init_mod = glm(Defaulter ~ ., family = binomial, data = smoteenn_cr)
print(summary(init_mod))
options(warn=-1)
pred = predict(init_mod, newdata = val, type = 'response')

#threshold = 0.16
pred1 = (ifelse(pred > 0.10, 1, 0))
#print(pred1)
#Checking the sensitivity, precision and confusion matrix
#pred1 = (pred1)
#Confusion matrix
#print(table(validation$Defaulter, pred1))
cm = (caret::confusionMatrix(as.factor(pred1),as.factor(val$Defaulter),positive = '1'))
print(cm)
pred1 = factor(pred1)
#Sensitivity
recall = cm[[4]][1]
recall
#Specificity
specifity = cm[[4]][2]
specifity
#Precision
precision = cm[[4]][5]
precision
#F1 score
F1 <- ((2 * precision * recall) / (precision + recall))
F1
#Youden's Index (Sensitivity + Specificity - 1)
# youdensIndex(valid$target, pred1)

#Mis-classification Error
#misClassError(val$target, pred)
library(ROCR)
ROCpred <- ROCR::prediction(pred,val$Defaulter)
ROCperf <- ROCR::performance(ROCpred,"tpr","fpr")
plot(ROCperf)
plot(ROCperf, colorize=T,
     print.cutoffs.at=seq(0,1,0.1), text.adj=c(-0.2,1.7))
perf_AUC=ROCR::performance(ROCpred,"auc") #Calculate the AUC value
AUC=perf_AUC@y.values[[1]]
AUC


###################### DECISION TREE ###################################


eval_mat = matrix()
#Building logisitic regression model on the  data  


library(rpart)
# Feature Scaling
classifier = rpart(formula = Defaulter ~ .,
                   data = smoteenn_cr,method = 'class')

# Predicting the Test set results
pred = predict(classifier, newdata =val, type = 'class')

#Confusion matrix
cm = caret::confusionMatrix(pred,as.factor(val$Defaulter),positive = '1')
print(cm)
#tpr, tn

#Sensitivity
recall = cm[[4]][1]

#Specificity
specifity = cm[[4]][2]

#Precision
precision = cm[[4]][5]

#F1 score
F1 <- ((2 * precision * recall) / (precision + recall))

#Youden's Index (Sensitivity + Specificity - 1)
#youdensIndex(validation$Defaulter, pred1)


#options(warn=1)

#print(eval_mat)
#plot(classifier)
#text(classifier)
library(ROCR)
ROCpred <- ROCR::prediction(as.numeric(pred), val$Defaulter)

perf_AUC=ROCR::performance(ROCpred,"auc") #Calculate the AUC value
AUC=perf_AUC@y.values[[1]]
AUC
perf_ROC=ROCR::performance(ROCpred,"tpr","fpr") #plot the actual ROC curve
plot(perf_ROC, main="ROC plot")
text(0.5,0.5,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
auc = AUC


eval_mat = c(recall,specifity,precision,F1,auc)
names(eval_mat) = c("recall","specificity","precision","F1","AUC")
eval_mat



############################ Naive Bayes #########################################

install.packages(naivebayes)
library(naivebayes)
eval_mat = matrix()

train = na.omit(train)
val = na.omit(val)

dim(train)
summary(pred)

set.seed(123)
library(e1071)
#library(klaR)
classifier_nb = naive_bayes(y = as.factor(smoteenn_cr$Defaulter), x = smoteenn_cr)
classifier_nb


pred = predict(classifier_nb, newdata = val[-1], type = 'class')
dim(pred)
pred
cm = table(pred,val$Defaulter)
cm1 = caret::confusionMatrix(pred,as.factor(val$Defaulter),positive = '1')
print(cm1)#tpr, tn

#Sensitivity
recall = cm1[[4]][1] 
recall
#Specificity
specifity = cm1[[4]][2]
specifity
#Precision
precision = cm1[[4]][5]
precision
#F1 score
F1 <- ((2 * precision * recall) / (precision + recall))
F1
#Youden's Index (Sensitivity + Specificity - 1)
#youdensIndex(validation$Defaulter, pred1)

library(ROCR)
#predictions= pred$posterior[,1]
pred=ROCR::prediction(as.numeric(pred),as.numeric(val$Defaulter))

perf_AUC=ROCR::performance(pred,"auc") #Calculate the AUC value
AUC=perf_AUC@y.values[[1]]
AUC
perf_ROC=ROCR::performance(pred,"tpr","fpr") #plot the actual ROC curve
plot(perf_ROC, main="ROC plot")
text(0.5,0.5,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
#options(warn=1)
eval_mat = c(recall,specifity,precision,F1,AUC)
names(eval_mat) = c("recall","specificity","precision","F1","AUC")
eval_mat




############################ Random Forest #########################################

eval_mat = matrix()
#Building logisitic regression model on the  data  



set.seed(123)
library(randomForest)
classifier = randomForest(Defaulter~.,data = smoteenn_cr,mtry=2, ntree=200)

# Predicting the Test set results
pred = predict(classifier, newdata = val[-1], type='class')
pred
#pred1 = (ifelse(pred > 0.10, 1, 0))
#pred1
cm2 = caret::confusionMatrix(as.factor(pred),as.factor(val$Defaulter))
print(cm2)

#Sensitivity
recall = cm2[[4]][1] 
recall
#Specificity
specifity = cm2[[4]][2]
specifity
#Precision
precision = cm2[[4]][5]
precision
#F1 score
F1 <- ((2 * precision * recall) / (precision + recall))
F1

library(ROCR)
#predictions= pred$posterior[,1]
pred=ROCR::prediction(as.numeric(pred),as.numeric(val$Defaulter))

perf_AUC=ROCR::performance(pred,"auc") #Calculate the AUC value
AUC=perf_AUC@y.values[[1]]
AUC
perf_ROC=ROCR::performance(pred,"tpr","fpr") #plot the actual ROC curve
plot(perf_ROC, main="ROC plot")
text(0.5,0.5,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
#options(warn=1)
eval_mat = c(recall,specifity,precision,F1,AUC)
names(eval_mat) = c("recall","specificity","precision","F1","AUC")
eval_mat


