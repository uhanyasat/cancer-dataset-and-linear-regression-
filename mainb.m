load cancer_dataset
[X,d] = cancer_dataset; %Type help cancer_dataset for more info

w=X'\d(2,:)'; %Training/MSE linear model creation
XX=X';
y=X'*w; %Activation/testing

[X,Y,T,AUC] = perfcurve(d(2,:),y',1);
figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC, AUC=' num2str(AUC) ])
YY=y;

train_X=XX(1:350,:);
train_Y=YY(1:350,:);

w=train_X\d(2,1:350)'; %Training/MSE linear model creation
y=train_Y; %Activation/testing
size(w);
size(y);
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);
print('Train Data Set')
AUC
figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['Train Data Set 2D ROC, AUC=' num2str(AUC) ])
test_X=XX(351:699,:);
test_Y=YY(351:699,:);
w=test_X\d(2,351:699)'; %Training/MSE linear model creation
y=test_Y; %Activation/testing
size(w);
size(y);
[X,Y,T,AUC] = perfcurve(d(2,351:699),y',1);
print('Test Data Set')
AUC
figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['Test Data Set 2D ROC, AUC=' num2str(AUC) ])

mdl = fitlm(train_X,train_Y)
ypred = predict(mdl,test_X);
errs = ypred - test_Y;
figure
histogram(errs)
title("Histogram of residuals - test data")