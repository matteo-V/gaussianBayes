
# Homework 2 - Naive Bayes Classification
## Matteo Vaiente | January 26, 2018
### Part 1 - Tuning Laplace parameter for text analysis
The aim of this analysis is to train a Naive Bayes classifier and use this for sentiment analysis on a set of data with binary outcomes, ham and spam. We use the data given on the [course webpage](http://www.rob-mcculloch.org/data/sms_spam.csv). <p>
    The code is parallelized to exploit multicore architectures. All models were run on ASU's research computing cluster [Saguaro](https://rcstatus.asu.edu/saguaro/index.php). We are going to need several packages for the analysis. These packages must be installed into user's library environment prior to running analysis. 

**Note**: R Packages must be installed using ```install.packages(packagename)```


```R
#import data wrangling packages
library(dplyr)
library(reshape2)
#import text processing packages
library(tm)
library(SnowballC)
#import read_csv
library(readr)
#import classifer
library(e1071)
#import parallelization library
library(parallel)
#import graphics
library(ggplot2)

```

    
    Attaching package: ‘dplyr’
    
    The following objects are masked from ‘package:stats’:
    
        filter, lag
    
    The following objects are masked from ‘package:base’:
    
        intersect, setdiff, setequal, union
    
    Loading required package: NLP
    
    Attaching package: ‘ggplot2’
    
    The following object is masked from ‘package:NLP’:
    
        annotate
    


#### A. Import and clean data
**Note:** Prior to analysis data must be copied into the same directory as this notebook using 
```bash 
curl http://www.rob-mcculloch.org/data/sms_spam.csv > sms_spam.csv
```


```R
#read in data
smsRaw <- read_csv("sms_spam.csv",
                   col_types = #define column types for read in
                     cols(
                       type = col_factor(levels=c('ham', 'spam')), #define factor levels
                       text = col_character() ) #message data
)
#pretty print (readr) and explore data
head(smsRaw)
```


<table>
<thead><tr><th scope=col>type</th><th scope=col>text</th></tr></thead>
<tbody>
	<tr><td>ham                                                                                                                                                              </td><td>Hope you are having a good week. Just checking in                                                                                                                </td></tr>
	<tr><td>ham                                                                                                                                                              </td><td>K..give back my thanks.                                                                                                                                          </td></tr>
	<tr><td>ham                                                                                                                                                              </td><td>Am also doing in cbe only. But have to pay.                                                                                                                      </td></tr>
	<tr><td>spam                                                                                                                                                                                                           </td><td><span style=white-space:pre-wrap>complimentary 4 STAR Ibiza Holiday or &lt;U+00A3&gt;10,000 cash needs your URGENT collection. 09066364349 NOW from Landline not to lose out! Box434SK38WP150PPM18+     </span></td></tr>
	<tr><td>spam                                                                                                                                                             </td><td>okmail: Dear Dave this is your final notice to collect your 4* Tenerife Holiday or #5000 CASH award! Call 09061743806 from landline. TCs SAE Box326 CW25WX 150ppm</td></tr>
	<tr><td>ham                                                                                                                                                              </td><td>Aiya we discuss later lar... Pick u up at 4 is it?                                                                                                               </td></tr>
</tbody>
</table>



Now we clean the textual data by converting to lowercase, removing numbers and stopwords for the target language, stemming and tokenization of word fragments, and removal of extra whitespace. 


```R
#create in memory corpus from vector of text in R
smsC = VCorpus(VectorSource(smsRaw$text))

# Clean and process text data
smsCC <- tm_map(smsC, content_transformer(tolower))
smsCC <- tm_map(smsCC, removeNumbers) # remove numbers
smsCC <- tm_map(smsCC, removeWords, stopwords()) # remove stop words
smsCC <- tm_map(smsCC, removePunctuation) # remove punctuation
smsCC <- tm_map(smsCC, stemDocument)
smsCC <- tm_map(smsCC, stripWhitespace) # eliminate unneeded whitespace

#create DTM data structure
smsDTM <- DocumentTermMatrix(smsCC)
```

#### B. Defining Model Functions
Now, we define the functions that we will use for training the classifier and evaluating output


```R
START BINARY CONVERSION FUNCTION
#create function to map counts to binary values
convertCounts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

### START TRAIN MODEL FUNCTION
# @parameters laplaceParam - tuning parameter for laplace model
trainNB <- function(laplaceParam, trainX, trainY, testX, testY){
  #train model with laplace smoothing parameter
  smsNB = naiveBayes(trainX, trainY, laplace=laplaceParam)
  #get model predictions
  yhat <- predict(smsNB, testX)
  #tabulate confusion matrix
  res <- table(yhat, testY)

  #calculate misclassification rate
  # calculated as sum off diagonals over total observations
  missRate <-  ( sum(res) - sum(diag(res)) ) / sum(res)
  missRate
}

## START per Sample runNB function
## to be used with lapply
## for each list element (vector) generates a sample data set and
runNB <- function(laplaceVals){ # each list element is a vector
  #generate sample set indicies
  sampSet <- sample(1:nObs, size = nTrain)
  #Create train splits
  smsTrain.X <- smsDTM[sampSet,]
  smsTrain.Y <- smsRaw[sampSet,]$type
  #create test splits
  smsTest.X <- smsDTM[-sampSet, ] #select those not in sample set
  smsTest.Y <- smsRaw[-sampSet, ]$type

  #find frequent words
  smsFreqWords = findFreqTerms(smsTrain.X, 15) #words that appear at leat 5 times

  #use filtered data with at least 15 counts for model
  smsFreqTrain = smsTrain.X[ , smsFreqWords]
  smsFreqTest = smsTest.X[ , smsFreqWords]

  #convert counts to binary indicators
  smsTrain = apply(smsFreqTrain, MARGIN = 2, convertCounts)
  smsTest = apply(smsFreqTest, MARGIN = 2, convertCounts)

  ## now train NB for each value in laplaceVals vector using the sample
  sapply(laplaceVals, trainNB, trainX=smsTrain, trainY=smsTrain.Y, testX=smsTest, testY=smsTest.Y)

}
```

#### C. Defining classifier model evaluation and tuning parameters
Now, we will tune the model over $N$ training test splits and $l$ values of the laplace grid. We define the laplace parameter values $l$ as an evenly spaced vector (incremented by $0.1$) in the interval from $[0, 1.5]$. 


```R
#Now define parameters data sets for training and testing
trainPercent <- 0.70
#get number of observations
nObs <- length(smsRaw$text)
#define number of training examples
nTrain <- floor(nObs * trainPercent)
#choose laplace parameters used to train model
laplGrid <- seq(0,1.5,.1)
#define number of simulations
N <- 10
#create data grid
y <- list(laplGrid)
x <- rep(y, N)
```

#### D. Train classifier
We will use parallelization to train the classifier to reduce computational time by splitting each training and test split to a seperate core. 


```R
#now run the experiments in sequential mode, slow right?
#cat("Running classifier on single core, 2 replicates")
#system.time( out <- lapply(x, runNB ) )

#### Solution: parallelize the code
## since we wrote the functions using the apply functions, the parallelization is trivial

#find number of cores
numCores <- parallel::detectCores()
print(numCores)
#set up cluster with fork to sahre variable addresses
#this should improve memory efficiency
clust <- parallel::makeCluster(numCores, type = "FORK")
#Set RNG for different streams
#This is necessary so cores do not replicate the same experiment
parallel::clusterSetRNGStream(cl=clust, iseed=NULL)
#run experiment on numCores
cat("Running experiment on", numCores, "cores, 2 replicates")
system.time( out2 <- parLapply(cl=clust, x, runNB) )
#stop the cluster
parallel::stopCluster(cl=clust)

#Now manipulate data for plotting
#melt results into data frame with factors as experiment results into data frame
res2 <- melt(out2)

#create laplace levels for each experiment and combien data
l <- rep(unlist(y), N )
res2 <- cbind(res2, l)
#write classifier evaluation data to file
outfile <- "classifierOutput.csv"
write_csv(res2, outfile)
```

#### E. Plot classifier performance
We visulalize the missclassification rate of the Naive Bayes classifier averaged over several training and test splits. Here we show the output for $N=10$ and $N=25$ replicates



```R
#source plotting function
source('plotNB.R')
#plot output
out.plot <- plot_NB(outfile)
##output plot
outplot <- "missclassPlot.png"
png(outplot)
print(outplot)
dev.off()
#save plot to output fule
ggsave(output, dpi=300)
```

![](classifierPlot_10.png)

![](classifierPlot_25.png)
<br>
**Note**: Laplace parameter grid resolution was increased

### Part 2 - Classification with numeric $x$

In our text classification we ended up have each $x_i$ binary and a $y$ binary.
What would we do if $x$ is numeric?
A common approach is to assume each $x_i$ is conditionally normal.
<br>
$$
X_i | Y=y \sim N(\mu_{iy},\sigma^2_{iy})
$$
<br> Recall the normal density is given by 
$$ 
f(x) = (2\pi\sigma^2)^{-\frac{1}{2}} \exp\Big[ -\frac{(x-\mu)^2}{2\sigma^2} \Big] 
$$
<br> 

We want to know the $\mathbb{P}(Y=y \mid x)$. We can start with Bayes' Rule which states the posterior probability is proportional to the prior times the likelihood$$ \mathbb{P}(Y=y \mid x) \propto \mathbb{P}(Y=y) \times f(x \mid Y=y) $$


We define the odds ratio $$
odds = \frac{\mathbb{P}(Y=1)}{\mathbb{P}(Y=0)} \, \frac{f(x \mid Y=1)}{f(x \mid Y=0)}
$$

Let's assume $f(x \mid Y=1) \sim N(\mu_1,\sigma^2)$ and 
$f(x \mid Y=0) \sim N(\mu_0,\sigma^2)$. That is, the variables are uncorrelated, with the same variance, so we can use the Naive Bayes approach. 

We define the log odds to be the log of the posterior odds. 
$$
\ell(odds) = log\Big{[}\frac{\mathbb{P}(Y=1)}{\mathbb{P}(Y=0)}
                  \frac{f(x \mid Y=1)}{f(x \mid Y=0)}\Big{]} 
$$


$$
\ell(odds) = log\Big{[}\frac{\mathbb{P}(Y=1)}{\mathbb{P}(Y=0)}\Big{]} + log\Big{[}\frac{f(x \mid Y=1)}{f(x \mid Y=0)}\Big{]}
$$

$$
\ell(odds) = log\Big{[}\frac{\mathbb{P}(Y=1)}{\mathbb{P}(Y=0)}\Big{]} + log\Big{[}f(x \mid Y=1)\Big{]} - log\Big{[}{f(x \mid Y=0)}\Big{]}
$$

$$
\ell(odds) = log\Big{[}\mathbb{P}(Y=1)\Big{]} - log\Big{[}\mathbb{P}(Y=0)\Big{]} + log\Big{[}f(x \mid Y=1)\Big{]} - log\Big{[}{f(x \mid Y=0)}\Big{]}
$$

By taking the log of the Normal density we obtain $ \ell(x) $
<p>
$$
\ell(x) = -\frac{1}{2}log(2\pi) - \frac{1}{2}log(\sigma^2) -\frac{1}{2\sigma^2}(x-\mu_y)^2
$$

Substituting the expression for $\ell(x)$ for each $log\Big{[}f(x \mid Y=1)\Big{]} $ and $log\Big{[}f(x \mid Y=0)\Big{]}$ replacing with the appropriate mean

$$
\ell(odds) = log\Big{[}\mathbb{P}(Y=1)\Big{]} - log\Big{[}\mathbb{P}(Y=0)\Big{]} -\frac{1}{2}log(2\pi) - \frac{1}{2}log(\sigma^2) -\frac{1}{2\sigma^2}(x-\mu_1)^2 +\frac{1}{2}log(2\pi) + \frac{1}{2}log(\sigma^2) +\frac{1}{2\sigma^2}(x-\mu_y)^2
$$

Notice that the terms containing $log(2\pi)$ and $log(\sigma^2)$ cancel and we obtain

$$
\ell(odds) = log\Big{[}\mathbb{P}(Y=1)\Big{]} - log\Big{[}\mathbb{P}(Y=0)\Big{]} -\frac{1}{2\sigma^2}(x-\mu_1)^2 +\frac{1}{2\sigma^2}(x-\mu_0)^2
$$

We want the rule to predict $Y=1$ when $ odds > 1 $. <p>
For the log odds this is $ 0 < \ell(odds)$

$$ 
0 < log\Big{[}\mathbb{P}(Y=1)\Big{]} - log\Big{[}\mathbb{P}(Y=0)\Big{]} -\frac{1}{2\sigma^2}(x-\mu_1)^2 +\frac{1}{2\sigma^2}(x-\mu_0)^2
$$

Combining prior probabilities and factoring by common terms gives us 
<p>
$$
 0 < log\Big{[}\frac{\mathbb{P}(Y=1)}{\mathbb{P}(Y=0)}\Big{]} + \frac{(x-\mu_0)^2}{2\sigma^2}-\frac{(x-\mu_1)^2}{2\sigma^2}  
$$

### Decision rule given numeric $x$
We can exponentiate each side to obtain a simple final rule for classification based on numeric $x$. This rule states that we predict $Y=1$ when the following inequality holds for the odds


$$
1  < \frac{\mathbb{P}(Y=1)}{\mathbb{P}(Y=0)} + \exp\Big{[} \frac{1}{2\sigma^2}\big{(}(x-\mu_0)^2-(x-\mu_1)^2\big{)} \Big{]}
$$

Similarly, we predict $Y=1$ when the following inequality holds for $\ell(odds)$, the log odds 

$$
0 < log \Big{[}\frac{\mathbb{P}(Y=1)}{\mathbb{P}(Y=0)}\Big{]} + \frac{(x-\mu_0)^2}{2\sigma^2}-\frac{(x-\mu_1)^2}{2\sigma^2} 
$$

Inuitively, this rule makes sense. As $x  \rightarrow \mu_0$, the first term tends to $0$. This leaves the negative term which shifts our prediction toward $Y=0$. However, if $x \rightarrow \mu_1$, we are left with a positive term shifting our prediction toward $Y=1$. In the extreme case that $ x \rightarrow \mu_0,\mu_1 $ our prediction is determined solely by the log prior odds.  
