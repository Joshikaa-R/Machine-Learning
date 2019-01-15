#libraries used
library(CatEncoders)
library(usdm)
library(tictoc)
library(car)
library(Metrics)
library(MLmetrics)
#logistic regression with gradient descent
#sigmoid function
sigmoid<-function(x)
{
    ans=1/(1+exp(-x))
    return(ans)
}

# batch gradient descent
gradient_descent<-function(feature,target,alpha,no_of_iterations)
{
feature=data.matrix(cbind(1,feature))
theta=rep(0,dim(feature)[2])
m=dim(feature)[1]
cost_grad=rep(0,no_of_iterations)
for( i in 1:no_of_iterations)
{
    yhat=sigmoid(feature%*%theta)
    cost_grad[i]=(1/m)*sum((-target*log(yhat)) - ((1-target)*log(1-yhat)))
    theta=theta-(alpha*((t(feature)%*%(yhat-target))/m))
}
plot(cost_grad[cost_grad!=0],main="batch gradient descent",ylab="cost")
return(theta)
}

#stochastic gradient descent implementation
stochastic_gradient_descent<-function(feature,target,alpha,no_of_iterations)
{
feature=data.matrix(cbind(1,feature))
#parameters
theta=rep(0,dim(feature)[2])
m=dim(feature)[1]
cost_sgd_avg=rep(0,no_of_iterations)
cost_sgd=rep(0,m)
k=1
for( i in 1:no_of_iterations)
{
    for( j in 1:m)
    {
        yhat=sigmoid(feature[j,]%*%theta)
        cost_sgd[k]=(1/m)*sum((-target[j]*log(yhat)) - ((1-target[j])*log(1-yhat)))
        k=k+1
        theta=theta-(alpha*c(yhat-target[j])*(feature[j,]))
    }
    if(i%%2==0) cost_sgd_avg[i]=sum(tail(cost_sgd,n=1000))/1000
}
plot(cost_sgd_avg[cost_sgd_avg!=0],main='stochastic gradient descent',ylab='cost')
return(theta)
}

#mini batch gradient descent implementation
minibatch_gradient_descent<-function(feature,target,alpha,no_of_iterations,batch_size)
{
feature=data.matrix(cbind(1,feature))
#parameters
theta=rep(0,dim(feature)[2])
m=dim(feature)[1]
cost_mgd_avg=rep(0,no_of_iterations)
cost_mgd=rep(0,m)
j=1
k=1
for( i in 1:no_of_iterations)
{  
    while(j+batch_size-1<m) {
        feature_mgd=sigmoid(feature[j:(j+batch_size-1),])
        #if(i==1) print(feature_mgd)
        target_mgd=target[j:(j+batch_size-1)]
        yhat=feature_mgd%*%theta
        cost_mgd[k]=(1/m)*sum((-target_mgd*log(yhat)) - ((1-target_mgd)*log(1-yhat)))
        k=k+1
        theta=theta-(alpha*((t(feature_mgd))%*%(yhat-target_mgd))/batch_size)
        j=j+batch_size
        }
        j=0
    if(i%%2==0) cost_mgd_avg[i]=sum(tail((cost_mgd[cost_mgd!=0]),n=100))/100
}
plot(cost_mgd_avg[cost_mgd_avg!=0],ylab="cost",main="mini batch gradient descent")
return(theta)
}


#prediction
predict_output<-function(theta,feature)
{
    feature=data.matrix(cbind(1,feature))
    y_predict=feature%*%theta
    ans=sigmoid(y_predict)
    ans[ans<0.5]=0
    ans[ans>=0.5]=1
    return(ans)
}


#reading data
data=read.csv("../input/german_credit_data.csv")
attach(data)

#looking the type of data
names(data)
str(data)

#missing values
sum(is.na(Age)) #0
sum(is.na(Sex)) #0
sum(is.na(Job)) #0
sum(is.na(Housing)) #0
sum(is.na(Saving.accounts)) #183
sum(is.na(Checking.account)) #394
sum(is.na(Credit.amount))  #0
sum(is.na(Duration)) #0
sum(is.na(Purpose))  #0

#data cleaning
data['X']=NULL
encode_sex=LabelEncoder.fit(Sex)            #female-1 male-2
data['Sex']=transform(encode_sex,Sex)
encode_housing=LabelEncoder.fit(Housing)    # free-1 own-2 rent-3
data['Housing']=transform(encode_housing,Housing)
saving.account_temp=na.omit(Saving.accounts)
encode_saving.accounts=LabelEncoder.fit(saving.account_temp)    # 
data['Saving.accounts']=transform(encode_saving.accounts,Saving.accounts)
checking.account_temp=na.omit(Checking.account)
encode_checking.account=LabelEncoder.fit(checking.account_temp)    # 
data['Checking.account']=transform(encode_checking.account,Checking.account)
encode_purpose=LabelEncoder.fit(Purpose)    
data['Purpose']=transform(encode_purpose,Purpose)
encode_risk=LabelEncoder.fit(Risk)    
data['Risk']=transform(encode_risk,Risk)
data$Risk[data$Risk==2]=0
attach(data)

#replacing missing values
avg <- round(mean(Saving.accounts, na.rm=TRUE))
indx <- which(is.na(Saving.accounts), arr.ind=TRUE)
Saving.accounts[indx] <- avg

avg <- round(mean(Checking.account, na.rm=TRUE))
indx <- which(is.na(Checking.account), arr.ind=TRUE)
Checking.account[indx] <- avg
data=data.frame((scale(data[,1:9])),data[,10])
attach(data)

#train test split
smp_size <- floor(0.95 * nrow(data))
set.seed(1939)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]
feature=train[,1:dim(train)[2]-1]
target=train[,dim(train)[2]]


#gradient descent
alpha=0.15
no_of_iterations=1000
tic()
theta=gradient_descent(feature,target,alpha,no_of_iterations)
run_time=toc()
grad_run_time=run_time$toc-run_time$tic
prediction=predict_output(theta,test[,1:9])
error_grad=LogLoss(test[,10],prediction)

#stochastic gradient descent
alpha=0.0005
no_of_iterations=300
tic()
theta=stochastic_gradient_descent(feature,target,alpha,no_of_iterations)
run_time=toc()
sgrad_run_time=run_time$toc-run_time$tic
prediction=predict_output(theta,test[,1:9])
error_sgrad=LogLoss(test[,10],prediction)


#minibatch gradient descent
alpha=0.005
no_of_iterations=1100
batch_size=200
tic()
theta=minibatch_gradient_descent(feature,target,alpha,no_of_iterations,batch_size)
run_time=toc()
mgrad_run_time=run_time$toc-run_time$tic
prediction=predict_output(theta,test[,1:9])
error_mgrad=LogLoss(test[,10],prediction)

#summary
model_type=c('gradient_descent','stochastic gradient descent','minibatch gradient descent')
error=c(error_grad,error_sgrad,error_mgrad)
run_time=c(grad_run_time,sgrad_run_time,mgrad_run_time)
table=data.frame(model_type,error,run_time)
table
