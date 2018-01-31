
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Naive Bayes Classification
## Extension to numeric features
### By: Matteo Vaiente 



We have seen the application of Naive Bayes classifier where we model the classification problem with a binary input feature (the so-called "bag-of-words" model). Here we will show how it can be extended to accompany numeric features. 


### But first... a little theory. 

A common approach is to assume each $$x_i$$ is conditionally normal.
<br>
$$
X_i | Y=y \sim N(\mu_{iy},\sigma^2_{iy})
$$
<br> Recall the normal density is given by 
$$ 
f(x) = (2\pi\sigma^2)^{-\frac{1}{2}} \exp\Big[ -\frac{(x-\mu)^2}{2\sigma^2} \Big] 
$$
<br> 

We want to know the $$\mathbb{P}(Y=y \mid x)$$. We can start with Bayes' Rule which states the posterior probability is proportional to the prior times the likelihood $$ \mathbb{P}(Y=y \mid x) \propto \mathbb{P}(Y=y) \times f(x \mid Y=y) $$


We define the odds ratio $$
odds = \frac{\mathbb{P}(Y=1)}{\mathbb{P}(Y=0)} \, \frac{f(x \mid Y=1)}{f(x \mid Y=0)}
$$
Let's assume $$f(x \mid Y=1) \sim N(\mu_1,\sigma^2)$$ and 
$$f(x \mid Y=0) \sim N(\mu_0,\sigma^2)$$. That is, the variables are uncorrelated, with the same variance, so we can use the Naive Bayes approach. 

We define the $\ell(odds)$ to be the log of the posterior odds. 
$$
\ell(odds) = log\Big{[}\frac{\mathbb{P}(Y=1)}{\mathbb{P}(Y=0)}
                  \frac{f(x \mid Y=1)}{f(x \mid Y=0)}\Big{]} 
$$
We want the rule to predict $$Y=1$$ when $$ odds > 1 $$. <p>
So for the log odds we want: $$ 0 < \ell(odds)$$

### Decision rule given numeric $$x$$
Using the normal distribution for the likelihood, we can obtain a simple final rule for classification based on numeric $$x$$. This rule states that we predict $$Y=1$$ when the following inequality holds

$$
 0 < log\Big{[}\frac{\mathbb{P}(Y=1)}{\mathbb{P}(Y=0)}\Big{]} + \frac{(x-\mu_0)^2}{2\sigma^2}-\frac{(x-\mu_1)^2}{2\sigma^2}  
$$


```R
#load data wrangling tools
suppressMessages(library(dplyr))
```


```R
#set RNG
set.seed(3825)

#define gaussian distribution parameters
sigma <- 1
mean0 <- 7.5
mean1 <- 10

#define number of data points per class
N <- 100
```


```R
#function to generate data
genData <- function(elem, mean, sd){
  dat <- rnorm(1, mean, sd)
  return(data.frame(elem, dat))
}

#simulate data
y0 <- lapply(rep(0, N), function(x) genData(elem=x, mean=mean0, sd=sigma))
y1 <- lapply(rep(1, N), function(x) genData(elem=x, mean=mean1, sd=sigma))

#combine into dataframte
dat <- data.frame( do.call(rbind, c(y1, y0)) )
```


```R
#get conditional means
#we can also use population means since data was simulated
mu1 <- sum( filter(dat, elem==1) ) / N
mu0 <- sum( filter(dat, elem==0) ) / N

#define prior probs
prior1 <- filter(dat, elem==1) %>% count() / (2 * N) #get prior probability
prior0 <- 1 - prior1 #by law of total probability for binary outcome
```


```R
#define data sets
#you could add train test splits here
x <- dat$dat
y <- dat$elem
```

### We use the decision rule derived above for predicting $Y_i$ given a numeric $x$. 


```R
# Function to compute log posterior odds
# this is using MLE, we can also use "true value" because of simulation
rule <- function(x){

  log(prior1/prior0) + (x - mu0)^2 / (2*sigma^2) - (x - mu1)^2 / (2*sigma^2)
}

#Function predict class via log posterior odds inequality
gaussNB <- function(odds){

  if(odds>0){
    return(1)
  }
  else
    return(0)
}
```

### Now calculate posterior odds and predict class labels


```R
#predict on all x
yhat <- sapply( sapply( x, function(predictor) rule(predictor) ),
                function(odds) gaussNB(odds) ) ## eval gaussNB for each

#tabulate and print confusion matrix
resTable <- table(yhat, y)
cat("Confusion Matrix\n")
print(resTable)

#compute and print misclassification rate
misclassRate <- ( sum(resTable) - sum(diag(resTable)) ) / sum(resTable)
cat("\nMissclassification Rate: ", misclassRate)
```

    Confusion Matrix
        y
    yhat  0  1
       0 97 19
       1  3 81
    
    Missclassification Rate:  0.11
