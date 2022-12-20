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

mdl = fitlm(X,Y)
[mx,my]=size(XX);
[nx,ny]=size(YY);
X1=XX(1:139,:);
Y1=YY(1:139,:);
mdl1 = fitlm(X1,Y1)



w=X1\d(2,1:139)'; %Training/MSE linear model creation
y=Y1; %Activation/testing
[X,Y,T,AUC] = perfcurve(d(2,1:139),y',1);
figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['subset1 2D ROC, AUC=' num2str(AUC) ])



X2=XX(140:279,:);
Y2=YY(140:279,:);
mdl2 = fitlm(X2,Y2)
w=X2\d(2,140:279)'; %Training/MSE linear model creation
y=Y2; %Activation/testing
[X,Y,T,AUC] = perfcurve(d(2,140:279),y',1);
figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['subset2 2D ROC, AUC=' num2str(AUC) ])


X3=XX(280:419,:);
Y3=YY(280:419,:);
mdl3 = fitlm(X3,Y3)
w=X3\d(2,280:419)'; %Training/MSE linear model creation
y=Y3; %Activation/testing
[X,Y,T,AUC] = perfcurve(d(2,280:419),y',1);
figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['subset3 2D ROC, AUC=' num2str(AUC) ])

X4=XX(420:559,:);
Y4=YY(420:559,:);
mdl4 = fitlm(X4,Y4)
w=X4\d(2,420:559)'; %Training/MSE linear model creation
y=Y4; %Activation/testing
[X,Y,T,AUC] = perfcurve(d(2,420:559),y',1);
figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['subset4 2D ROC, AUC=' num2str(AUC) ])

X5=XX(560:mx,:);
Y5=YY(560:mx,:);
mdl5 = fitlm(X5,Y5)
w=X5\d(2,560:mx)'; %Training/MSE linear model creation
y=Y5; %Activation/testing
[X,Y,T,AUC] = perfcurve(d(2,560:mx),y',1);
figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['subset5 2D ROC, AUC=' num2str(AUC) ])