#dataset used: https://www.kaggle.com/mehdidag/black-friday

#libraries used
library(CatEncoders)
library(caTools)
library(tictoc)
library(car)
library(Metrics)

#batch gradient descent implementation
gradient_descent<-function(feature,target,alpha,lambd,no_of_iterations)
{
feature=data.matrix(cbind(1,feature))
#parameters
theta=rep(0,dim(feature)[2])
m=dim(feature)[1]
cost_grad=rep(0,no_of_iterations)
for( i in 1:no_of_iterations)
{
    yhat=feature%*%theta
    cost_grad[i]=sum((yhat-target)^2)/(2*m)
    theta=theta-(alpha*(((t(feature)%*%(yhat-target))/m)+((lambd/m)*theta)))
}
plot(cost_grad,main="batch gradient descent",ylab="cost")
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
        yhat=feature[j,]%*%theta
        cost_sgd[k]=((yhat-target[j])^2)/(2*m)
        k=k+1
        theta=theta-(alpha*c(yhat-target[j])*(feature[j,]))
    }
    if(i%%2==0) cost_sgd_avg[i]=sum(tail(cost_sgd,n=1000))/1000
}
plot(cost_sgd_avg[cost_sgd_avg!=0],main="stochastic gradient descent",ylab="cost")
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
        feature_mgd=feature[j:(j+batch_size-1),]
        #if(i==1) print(feature_mgd)
        target_mgd=target[j:(j+batch_size-1)]
        yhat=feature_mgd%*%theta
        cost_mgd[k]=sum((yhat-target_mgd)^2)/(2*batch_size)
        k=k+1
        theta=theta-(alpha*((t(feature_mgd))%*%(yhat-target_mgd))/batch_size)
        j=j+batch_size
        }
        j=0
    if(i%%2==0) cost_mgd_avg[i]=sum(tail((cost_mgd[cost_mgd!=0]),n=100))/100
}
plot(cost_mgd_avg[cost_mgd_avg!=0],main="minibatch gradient descent",ylab="cost")
return(theta)
}

#prediction
predict_output<-function(theta,feature)
{
    feature=data.matrix(cbind(1,feature))
    y_predict=feature%*%theta
    return(y_predict)
}

#data reading
data=read.csv("../input/BlackFriday.csv")
attach(data)

#data cleaning
data['User_ID']=NULL
data['Product_ID']=NULL

encode_gender=LabelEncoder.fit(Gender)
data['Gender']=transform(encode_gender,Gender)

encode_age=LabelEncoder.fit(Age)
data['Age']=transform(encode_age,Age)

encode_city_category=LabelEncoder.fit(City_Category)
data['City_Category']=transform(encode_city_category,City_Category)

encode_Stay_In_Current_City_Years=LabelEncoder.fit(Stay_In_Current_City_Years)
data['Stay_In_Current_City_Years']=transform(encode_Stay_In_Current_City_Years,Stay_In_Current_City_Years)

data['Product_Category_2']=NULL
data['Product_Category_3']=NULL

#data structure
str(data)
attach(data)

#correlation
correlation=cor(data)
data['Marital_Status']=NULL                # since correlation is very low
data['Stay_In_Current_City_Years']=NULL     # since correlation is very low

#train test split
smp_size <- floor(0.95 * nrow(data))
set.seed(1939)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]
feature=train[,1:dim(train)[2]-1]
target=train[,dim(train)[2]]

#regression fit using lm function
tic()
model=lm(target~data.matrix(feature))  #R-squared value is very small
error_lm=rmse(target,predict(model))
run_time=toc()
lm_run_time=run_time$toc-run_time$tic
plot(model, which = 1)                 #residual plot is almost random ->relationship between variables is linear

#regression fit using normal equation
x=data.matrix(cbind(1,feature))
y=matrix(target,nc=1)
tic()
theta_normal=solve(t(x)%*%x)%*%(t(x)%*%y)
prediction_normal=predict_output(theta_normal,test[,1:5])
run_time=toc()
norm_run_time=run_time$toc-run_time$tic
error_normal=rmse(test[,6],prediction_normal)

#regression fit using gradient descent
#hyper parameters
alpha=0.00005
no_of_iterations=10000
lambd=0.02
tic()
theta=gradient_descent(feature,target,alpha,lambd,no_of_iterations)
run_time=toc()
grad_run_time=run_time$toc-run_time$tic
prediction=predict_output(theta,test[,1:5])
error_grad=rmse(test[,6],prediction)

#stochastic gradient descent
alpha=0.00001
no_of_iterations=30
tic()
theta=stochastic_gradient_descent(feature,target,alpha,no_of_iterations)
run_time=toc()
sgrad_run_time=run_time$toc-run_time$tic
prediction=predict_output(theta,test[,1:5])
error_sgrad=rmse(test[,6],prediction)

#minibatch gradient descent
alpha=0.0001
no_of_iterations=300
batch_size=128
tic()
theta=minibatch_gradient_descent(feature,target,alpha,no_of_iterations,batch_size)
run_time=toc()
mgrad_run_time=run_time$toc-run_time$tic
prediction=predict_output(theta,test[,1:5])
error_mgrad=rmse(test[,6],prediction)

#summary
model_type=c('lm','normal equation','gradient_descent','stochastic gradient descent','minibatch gradient descent')
error=c(error_lm,error_normal,error_grad,error_sgrad,error_mgrad)
run_time=c(lm_run_time,norm_run_time,grad_run_time,sgrad_run_time,mgrad_run_time)
table=data.frame(model_type,error,run_time)
table