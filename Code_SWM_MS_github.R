# ADAPTIVE BOOSTING ####
# create empty matrices 
{predict_test_rf_class_outer<-NULL
multiResultClass <- function(result1=NULL,result2=NULL,result3=NULL,result4=NULL,
                             result5=NULL,result6=NULL,result7=NULL,result8=NULL){me <- list(
                               result1 = result1,
                               result2 = result2,
                               result3 = result3,
                               result4 = result4,
                               result5 = result5,
                               result6 = result6,
                               result7 = result7,
                               result8 = result8)
                             #Set the name for the class
                             class(me) <- append(class(me),"multiResultClass")
                             return(me)}

predict_all_outer<-NULL;predict_all_connect_all_outer<-NULL;predict_all_ensemble_all_outer<-NULL
varImp_all_outer<-NULL;varImp_SC_FC_all_outer<-NULL;varImp_connect_all_outer<-NULL
auc_outer_outerloop<-NULL; outer_auc_connect_all_outer<-NULL; outer_auc_ensemble_all_outer<-NULL
best_hyper_outer<-NULL; best_hyper_connect_outer<-NULL; best_hyper_ensemble_outer<-NULL
best_threshold_all_outer<-NULL;best_threshold_all_connect_all_outer<-NULL;best_threshold_all_ensemble_all_outer<-NULL
result_ConfMatrix_outer<-NULL;result_ConfMatrix_outer_connect_all_outer<-NULL;result_ConfMatrix_outer_ensemble_all_outer<-NULL
err_cv_outer<-NULL;err_cv_connect_outer<-NULL;err_cv_ensemble_outer<-NULL
dim_data_outer<-NULL;result_ConfMatrix_outer1<-NULL
auc_single_ensemble_outerloop<-NULL
result_ConfMatrix_outer05<-NULL
result_ConfMatrix_outer05_1<-NULL
result_ConfMatrix_outerAUC<-NULL
result_ConfMatrix_outerAUC_1<-NULL
aucpr_outer_outerloop<-NULL;brierscore_allmodels<-NULL;predicted_observed_brier_outerloop<-NULL
mean_pred_ensemble_outer12345_weighted<-NULL;predict_all_ensemble_all_outer_weighted<-NULL
predict_test_rf_confusion_outer<-NULL;varImp_all_outer1<-list();result_ConfMatrix_outerAUC<-NULL
cor_pred<-NULL
brierscoreouter<-NULL
}
# define the number of the folds and inner loop iteration number
OuterKfold<-5;InnerKfold<-5;InnerIterNumber<-5

# run in parallel
parallelnumber<-20
myCluster <- makeCluster(parallelnumber)
registerDoMC(parallelnumber)

# define the hyperparameter interval
minsplitinterval<-seq(10,50,10)
cpinterval<-c(0.1,0.01,0.001,0.0001)
numberofiter<-50

# Start outerloop ####

AdaBoost_Classification<-foreach(it=rep(1:parallelnumber,2),.combine = rbind,.multicombine=TRUE,.packages=c("nnet","rminer","caret","AUC","e1071","randomForest")) %dopar% {
                                                                                                    
for(outerloop in 1:OuterKfold){
  
  # Split the data in 10 partitions
  auc_outer_outerloop_onlyforthisouterloop<-NULL
  aucpr_outer_outerloop_onlyforthisouterloop<-NULL
  folds_outerloop<-createFolds(factor(as.data.frame(data_used1[[1]])$Output_class),k=OuterKfold,list = FALSE)  
  for(model in 1:length(data_used1)){
    
    cat(paste("model=",model))
    
    data_used<-data_used1[[model]]
    data_used<-as.data.frame(data_used)
    names1<-names(data_used)
    names(data_used)<-make.names(names1, unique = TRUE, allow_ = TRUE)
    data_used$Output_class<-as.factor(data_used$Output_class) 
    
    # Create a train dataset using 4 partitions over 5                                                           
    trainData <- data_used[folds_outerloop != outerloop, ]
    
    # Create a test dataset using 1 partition over 5                                                           
    testData <- data_used[folds_outerloop == outerloop, ]
    
    err_cv<-NULL
    # Start inner loop ####
    # Repeat CV 5 times for different partitions
    for (iterinner in 1:InnerIterNumber) {
      # cat(paste("iterinner=",iterinner))
      
      # Create partitions from train dataset
      folds_outerloop_inner<-createFolds(factor(trainData$Output_class),k=InnerKfold,list = FALSE)
      
      for(innerloop in 1:InnerKfold){
        
        trainingData <- trainData[folds_outerloop_inner != innerloop, ]
        validationData <- trainData[folds_outerloop_inner == innerloop, ]
        
        normParam_training <- preProcess(trainingData,method = c("center", "scale"))
        trainingData <- predict(normParam_training, trainingData)
        validationData <- predict(normParam_training, validationData)
        
        trainingData<-random.impute.data.frame(trainingData, 1:(dim(trainingData)[2]))
        validationData<-random.impute.data.frame(validationData, 1:(dim(validationData)[2]))
        
        # Inner loop CV for demographic
        
        for (ntree in minsplitinterval) {
          for (mtry in cpinterval) {
            
            # Fit the model using a couple of the hyperparameters
            control1<-rpart.control(minsplit = ntree, minbucket = round(ntree/3), cp = mtry, 
                                    maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, xval = 10,
                                    surrogatestyle = 0, maxdepth = 10)
            
            
            innermodel_rf_prob<-  adabag::boosting(Output_class ~ ., trainingData, boos = TRUE, mfinal = numberofiter, coeflearn = 'Breiman', 
                                                   na.action=na.roughfix,control=control1)
            
            
            # Predict the validation dataset
            predict_validation_rf_prob<-adabag::predict.boosting(innermodel_rf_prob,validationData,type="prob")$prob[,1]
            auc_inner <- ROSE::roc.curve(validationData$Output_class, predict_validation_rf_prob,plotit = FALSE)$auc
            err_cv<-rbind(err_cv,c(iterinner,innerloop,ntree,mtry,auc_inner,model))
            
          }
        }
        
      }
      
    }
    err_cv_outer<-rbind(err_cv_outer,err_cv)
    
    # End inner loop ####
    
    # Best hyperparam ####
    param_median_auc<-NULL
    for(ntreebest in levels(as.factor(err_cv[,3]))){
      for(mtrybest in levels(as.factor(err_cv[,4]))){
        row1<-which(err_cv[,3]==ntreebest)
        row2<-which(err_cv[,4]==mtrybest)
        param_median_auc<-rbind(param_median_auc,c(as.numeric(ntreebest),as.numeric(mtrybest),
                                                   as.numeric(median(err_cv[intersect(row1,row2),5]))))
      }
    }
    best_hyper<-c(param_median_auc[which.max(param_median_auc[,3]),1],
                  param_median_auc[which.max(param_median_auc[,3]),2])
    
    best_hyper_outer<-rbind(best_hyper_outer,best_hyper)
    
    # normalize ####
    normParam_train <- preProcess(trainData,method = c("center", "scale"))
    trainData <- predict(normParam_train, trainData)
    testData <- predict(normParam_train, testData)
    
    trainData<-random.impute.data.frame(trainData, 1:(dim(trainData)[2]))
    testData<-random.impute.data.frame(testData, 1:(dim(testData)[2]))
    
    control1<-rpart.control(minsplit = best_hyper[1], minbucket = round(best_hyper[1]/3), cp = best_hyper[2], 
                            maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, xval = 10,
                            surrogatestyle = 0, maxdepth = 10)
    
    outermodel_rf_prob<-  adabag::boosting(Output_class ~ ., trainData, boos = TRUE, mfinal = numberofiter, coeflearn = 'Breiman', 
                                           na.action=na.roughfix,control=control1)
    
    # Predict the test dataset
    predict_test_rf_prob<-adabag::predict.boosting(outermodel_rf_prob,testData,type="prob")$prob[,1]
    predict_test_rf_class<-adabag::predict.boosting(outermodel_rf_prob,testData,type="prob")$class
    predict_test_rf_confusion<-adabag::predict.boosting(outermodel_rf_prob,testData,type="prob")$confusion
    
    # confusion matrix ####
    result_ConfMatrixAUC<-caret::confusionMatrix(as.factor(predict_test_rf_class),testData$Output_class,positive="1")
    result_ConfMatrix_outerAUC<-cbind(result_ConfMatrix_outerAUC,result_ConfMatrixAUC)
    
    predict_test_rf_class_outer<-list(predict_test_rf_class_outer,cbind(predict_test_rf_class,
                                                                        as.numeric(as.character(testData$Output_class)),
                                                                        rep(model,length(predict_test_rf_class)),
                                                                        rep(outerloop,length(predict_test_rf_class))))
    
    predict_test_rf_confusion_outer<-rbind(predict_test_rf_confusion_outer,c(predict_test_rf_confusion[1],
                                                                             predict_test_rf_confusion[2],
                                                                             predict_test_rf_confusion[3],
                                                                             predict_test_rf_confusion[4]))
    ## var imp demo ####
    varImp_all_outer<-list(varImp_all_outer,as.numeric(outermodel_rf_prob$importance))
    
    # Predict&Output Demo ####
    predict_all_outer<-rbind(predict_all_outer,cbind(predict_test_rf_prob,
                                                     as.numeric(as.character(testData$Output_class)),
                                                     rep(model,length(predict_test_rf_prob)),
                                                     rep(outerloop,length(predict_test_rf_prob))))   
    
    # AUC for demo
    auc_outer <- ROSE::roc.curve(testData$Output_class, predict_test_rf_prob,plotit = FALSE)$auc
    require(PRROC)
    fg <- predict_test_rf_prob[testData$Output_class == 1]
    bg <- predict_test_rf_prob[testData$Output_class == 0]
    # PR Curve
    auc_pr_outer1 <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)  
    
    fg <- predict_test_rf_prob[testData$Output_class == 0]
    bg <- predict_test_rf_prob[testData$Output_class == 1]
    # PR Curve
    auc_pr_outer2 <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)  
    
    auc_outer_outerloop_onlyforthisouterloop<-rbind(auc_outer_outerloop_onlyforthisouterloop,c(auc_outer,auc_pr_outer1,auc_pr_outer2,model=model,outerloop=outerloop))
    auc_outer_outerloop<-rbind(auc_outer_outerloop,c(auc_outer,auc_pr_outer1,auc_pr_outer2,model))
    
  } ## end of the model loop
  
  
  # Data dimensions  
  dim_data<-c(dim(data_used)[1],
              dim(data_used)[2],
              length(which(data_used$Output_class==1)),
              length(which(data_used$Output_class==0)),
              length(which(testData$Output_class==1)),
              length(which(testData$Output_class==0)),
              length(which(trainData$Output_class==1)),
              length(which(trainData$Output_class==0)))
  dim_data_outer<-rbind(dim_data_outer,dim_data)
} ## end of the outer loop

# Save the classification results and variable importance in the same list
result <- multiResultClass()
result$result1 <- predict_all_outer
result$result2 <-best_threshold_all_outer
result$result3 <- auc_outer_outerloop
result$result4 <- best_hyper_outer
result$result5 <-result_ConfMatrix_outerAUC
result$result6 <- dim_data_outer
result$result7 <- list(err_cv_outer)
result$result8 <- varImp_all_outer

return(result) 
} # end of parallel loop

# load the results
results<-AdaBoost_Classification

# AUC ####
results_SWM_auc<-NULL
for(i in 101:200){ # use this interval for 100 outerloop
  results_SWM_auc<-rbind(results_SWM_auc,results[[i]])
}
dim(results_SWM_auc)

# create a list for the AUC results of 6 models
yalist<-list(
  as.numeric(results_SWM_auc[which(results_SWM_auc[,10]==1),1]),
  as.numeric(results_SWM_auc[which(results_SWM_auc[,10]==2),1]),
  as.numeric(results_SWM_auc[which(results_SWM_auc[,10]==3),1]),
  as.numeric( results_SWM_auc[which(results_SWM_auc[,10]==4),1]),
  as.numeric(results_SWM_auc[which(results_SWM_auc[,10]==5),1]),
  as.numeric( results_SWM_auc[which(results_SWM_auc[,10]==6),1]))

# present the summary of AUC results
AUCresults_mean_sd<-matrix(NA,nrow=6,ncol=5)
for(i in 1:6){
  AUCresults_mean_sd[i,1]<-mean(yalist[[i]])
  AUCresults_mean_sd[i,2]<-sd(yalist[[i]])
  AUCresults_mean_sd[i,3]<-median(yalist[[i]])
  AUCresults_mean_sd[i,4]<-summary(yalist[[i]])[2]
  AUCresults_mean_sd[i,5]<-summary(yalist[[i]])[5]
  
}
AUCresults_mean_sd<-as.data.frame(AUCresults_mean_sd)
names(AUCresults_mean_sd)<-c("average","sd","median","1st quartile","3rd quartile")

# perform wilcox test to compare AUC results between the models
for(i in 1:6){print(summary(yalist[[i]]))}
pairwise_AUC_comparison<-matrix(NA,6,6)
for(i in 1:6){
  for(k in 1:6){
    pairwise_AUC_comparison[i,k]<-wilcox.test(yalist[[i]],yalist[[k]])$p.value
    
  }
}

rownames(pairwise_AUC_comparison)<-c("SWM","DWM","SWM and DWM","Cortical thickness","rsFC","All variables")
colnames(pairwise_AUC_comparison)<-c("SWM","DWM","SWM and DWM","Cortical thickness","rsFC","All variables")
round(pairwise_AUC_comparison,3)
# correct p values for multiple comparison
pairwise_AUC_comparison_afterpcorrection<-p.adjust(pairwise_AUC_comparison,method="BH")
round(pairwise_AUC_comparison_afterpcorrection_matrix<-matrix(data = pairwise_AUC_comparison_afterpcorrection, 
                                                              nrow = 6,
                                                              ncol = 6,
                                                              byrow = TRUE),3)
rownames(pairwise_AUC_comparison_afterpcorrection_matrix)<-c("SWM","DWM","SWM and DWM","Cortical thickness","rsFC","All variables")
colnames(pairwise_AUC_comparison_afterpcorrection_matrix)<-c("SWM","DWM","SWM and DWM","Cortical thickness","rsFC","All variables")
round(pairwise_AUC_comparison_afterpcorrection_matrix,3)

# AUC figure ####
par(mar=c(10,6,4,2.1))
#sets the bottom, left, top and right margins respectively of the plot region in number of lines of text.
plot(0,0,type="n",xlim=c(0,7), ylim=c(0.5,1.2),  
     xaxt = 'n',yaxt = 'n', xlab =" ", ylab = "",  main ="AUC",cex.main=3,cex.lab=1.8)
axis(2,at=seq(0.5,1,0.05),labels=seq(0.5,1,0.05))
for(hline in seq(0.5,1,0.05)){
  abline(h = hline,lty=2)}
library("viridis")
col1 = magma(8)[3:8]
library(vioplot)
for (i in 1:6) { vioplot(na.omit(yalist[[i]]), at = i, add = T, col =col1[i] ) }
axis(side=1,las=3,at=1:6,
     labels=c("SWM","DWM","SWM and DWM","Cortical \n thickness","RSFC","All variables"),
     cex.axis=1.4)

segments(1,1.18,6,1.18,lwd=2)
segments(2,1.15,6,1.15,lwd=2)
segments(3,1.12,6,1.12,lwd=2)
segments(4,1.09,6,1.09,lwd=2)
segments(5,1.06,6,1.06,lwd=2)

#SWM
text(2,1.19, "*", cex = 1.5);text(4,1.19, "*", cex = 1.5);text(5,1.19, "*", cex = 1.5);text(6,1.19, "*", cex = 1.5)
#DWM
text(3,1.16, "*", cex = 1.5);text(4,1.16, "*", cex = 1.5);text(5,1.16, "*", cex = 1.5);text(6,1.16, "*", cex = 1.5)
#SWM and DWM
text(4,1.13, "*", cex = 1.5);text(5,1.13, "*", cex = 1.5);text(6,1.13, "*", cex = 1.5)
# Cort thickness
text(5,1.10, "*", cex = 1.5)


# variable importance  ####
results<-AdaBoost_Classification

results_varimp_swm<-NULL;results_varimp_dwm<-NULL;results_varimp_swm_dwm<-NULL;results_varimp_corthickness<-NULL;results_varimp_fc<-NULL;results_varimp_allvariables<-NULL
for(i in 601:700){ # use this loop if you run the outerloop 100 times
  results_varimp_swm<-cbind(results_varimp_swm,cbind(results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                     results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                     results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                     results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                     results[[i]][[1]][[1]][[1]][[1]][[1]][[2]])) #model SWM
  
  results_varimp_dwm<-cbind(results_varimp_dwm,cbind(results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                     results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                     results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                     results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                     results[[i]][[1]][[1]][[1]][[1]][[2]])) # model DWM
  
  
  results_varimp_swm_dwm<-cbind(results_varimp_swm_dwm,cbind(results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                             results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                             results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                             results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                             results[[i]][[1]][[1]][[1]][[2]])) #model SWM and DWM
  
  results_varimp_corthickness<-cbind(results_varimp_corthickness,cbind(results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                                       results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                                       results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                                       results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                                       results[[i]][[1]][[1]][[2]])) # model cortical thickness
  
  results_varimp_fc<-cbind(results_varimp_fc,cbind(results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                   results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                   results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                   results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                   results[[i]][[1]][[2]])) #model FC
  
  results_varimp_allvariables<-cbind(results_varimp_allvariables,cbind(results[[i]][[2]], # model all variables
                                                                       results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                                       results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                                       results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]],
                                                                       results[[i]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[1]][[2]]))
}
