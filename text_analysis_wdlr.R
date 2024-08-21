install.packages(c("quanteda", "topicmodels"))

library(quanteda)
library(topicmodels)

# when working with text data, we need to set stringsAsFactors as FALSE:
reviews = read.csv("product_reviews.csv", stringsAsFactors = FALSE)

# what columns are in our reviews data?
colnames(reviews)

# the reviews are in the "Content" column; this will call the first review:
reviews$Content[1:5]

# Create a corpus for those reviews, which just turns a text column into a set of documents:
review_corpus = corpus(reviews$Content[1:200])
summary(review_corpus)

review_tokens <- tokens(review_corpus[1:200], remove_punct = TRUE, remove_numbers = TRUE)

review_tokens_semi_clean <-tokens_remove(review_tokens, pattern = stopwords("en"))

review_tokens_clean<-tokens_wordstem(review_tokens_semi_clean)

# Summarize the text using a document-term frequency matrix (DFM):
review_dfm = dfm(review_tokens_clean)
print(review_dfm)


# "Trim" the DFM to keep only words that occur in more than 1% of documents but less than 75%:   
trimmed_dfm = dfm_trim(review_dfm, min_docfreq = 0.01, max_docfreq = 0.75, docfreq_type = "prop")
print(trimmed_dfm)
dim(trimmed_dfm)

## SENTIMENT ANALYSIS ---------------------------------------------------------------

# To run sentiment analysis, we need a lexicon. In Quanteda, lexicons are also referred 
# to as dictionaries. An easy lexicon to use is the LSD lexicon, which just scores words 
# as negative or positive. It is available simply by calling "data_dictionary_LSD2015":

data_dictionary_LSD2015

# To use this for sentiment analysis, we again use the dfm command:

senti = dfm_lookup(trimmed_dfm, dictionary = data_dictionary_LSD2015)
senti
# Let's compute the fraction of positive words:
frac_pos = senti[,2]/(senti[,1]+senti[,2])

# Does the fraction of positive words correlate with the star rating?
boxplot(as.matrix(frac_pos) ~ reviews$Rating[1:200], horizontal = TRUE, xlab = "Fraction of Positive Sentiment", ylab = "Star Rating")

## TOPIC MODELING ---------------------------------------------------------------

# First, convert the DFM to the format needed for LDA from the topicmodels package:
dtm_reviews = convert(trimmed_dfm, to="topicmodels")

# Pick the number of topics, and run LDA:
n_topics = 10
rev_lda = LDA(dtm_reviews, k=n_topics, method="Gibbs") 
# Note: This may take some time to run!
# There are two methods you can use, which control which algorithm is learn the model parameters:
# the default is method="VEM", which is faster, but the other option, method="Gibbs" often gives 
# better results. 

# To see the results, we have to either call "terms" or "posterior"
#  - "terms" shows us the top k words that are associated with each topic:
terms(rev_lda, k=5)

#  - "posterior" shows which topics a given document (or, in this case, review) features,
#    similar to the factor scores from factor analysis 
posterior(rev_lda)$topics[1,]

# Let's see if these topics are predictive of the star rating:
summary(lm(reviews$Rating[1:200] ~ 0 + posterior(rev_lda)$topics))

# IMPORTANT: you can't use an intercept here (ask yourself: why?)
