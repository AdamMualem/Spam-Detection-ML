# Spam Review Detection Using Machine Learning


## Problem:

The emergence of e-commerce has transformed the way we shop, trade, and connect with products and services. Consumers can purchase a variety of goods and services with just a few clicks. Online reviews have become a central element in the decision-making process for consumers. According to consumer research conducted by Bazzar of 30,000+ global shoppers, the majority (88%) of shoppers use reviews to discover and evaluate products (Byrne 2022). Reviews have the power to influence a brand’s trust and credibility.

This surge in online reviews has also led to the proliferation of fake and spam reviews, which can mislead consumers and harm businesses. Fake reviews influenced around $152 billion in global spending on lackluster products and services, according to a report from the World Economic Forum (Time 2021). The main issue is the erosion of trust in the e-commerce markets. With this happening this will lead to a major market decline with most online retailers. To maintain the integrity of online reviews and ensure consumers can make informed decisions, there's a pressing need for robust fake review detection methods. 


[Past literature](https://www.sciencedirect.com/science/article/pii/S0969698921003374?via%3Dihub#fig3) has highlighted the limitations of older detection methods, citing their diminished reliability. A notable breakthrough in this domain is showcased in this paper, where a model boasting a remarkable 96% accuracy rate was successfully developed. Despite these advancements, the continuous progress of GPT models suggests that the dependability of these methods continues to wane. Furthermore, recent [research](https://pubsonline.informs.org/doi/abs/10.1287/mksc.2022.1353?casa_token=jfFrBs_Y8PoAAAAA:WmkBe2iyVjJHHNF1H7K7Nz6hn-jKHwBnIBhI_9hsYLB8O3bJSx_-0DOHcODBolNoGHs3MjBKyAfQ) sheds light on the magnitude of this issue. A substantial portion of reviews on Amazon are fraudulent. This shows the critical need for advanced detection methods to safeguard the integrity of online platforms. Overall, my motivation is to create a comprehensive model that will perform better and to keep up with the adapting environment.

## Dataset

My study utilized a dataset containing 24,000 reviews, split between 20,000 original reviews (real) and 4,000 code-generated (fake) reviews. This dataset encompasses reviews from ten categories, reflecting a broad spectrum of consumer products and services.

![Alt text](Label.png?raw=true "Optional Title")
![Alt text](Categories.png?raw=true "Optional Title")
![Alt text](Ratings.png?raw=true "Optional Title")


## Data Preprocessing and Feature Selection

### Preprocessing

In  data preprocessing, I utilized Synthetic Minority Over-sampling Technique (SMOTE). SMOTE is an algorithm used to balance class distribution in a dataset. It creates synthetic samples from the minority class, as opposed to oversampling with replacement. This can help in stopping overfitting. I used SMOTE in the dataset for the code-generated class since I had a significant amount more of real reviews.

Additionally, I utilized the Natural Language Toolkit's (NLTK) Sentiment Intensity Analyzer to perform sentiment analysis. This step allowed me to obtain a sentiment score that quantifies the positivity or negativity of the language used. The sentiment analysis assesses the emotional tone behind words and provides a composite score known as a compound score. This compound score is a normalized, weighted composite score that takes into account the intensity of the sentiment expressed. 

- A compound score equal to or greater than 0.05 was a sign of positive sentiment and labeled as 1.
- A compound score less than or equal to -0.05 was a sign of a negative sentiment and labeled as -1.
- Scores that did not meet these thresholds were considered neutral and labeled as 0.

These scores were then added as a numerical feature to the dataset.

### Feature Selection Process

The feature selection process was informed by the hypothesis that certain characteristics of the review text can be indicative of its authenticity. To test this hypothesis, I engineered several features:

- **Sentiment Scores:**  As part of the feature engineering, I included the sentiment scores derived from the sentiment analysis. The sentiment of a review text is a valuable feature as fake reviews might systematically differ in sentiment compared to genuine ones.
![Alt text](sentiment_socre.png?raw=true "Optional Title")
- **Review Char Length:** The character length of reviews was computed and included as a feature under the assumption that the length could correlate with the review's authenticity. Fake reviews might be shorter or unusually long when compared to genuine reviews.
![Alt text](review_length.png?raw=true "Optional Title")
- **Spelling Errors Count:** I introduced a spelling error count feature by utilizing the phunspell library to identify and count misspellings within the text. This feature is predicated on the idea that fake reviews may exhibit different patterns of spelling errors.
![Alt text](spelling.png?raw=true "Optional Title")
- **Review Word Length:** *I added the number of words in each review for similar reasons as the character length and I hoped it might provide some additional insight along with character length. As you can see the two graphs have similar distributions.
![Alt text](word_length.png?raw=true "Optional Title")



These features were selected based on their potential to contribute meaningful signals for the classification task. The approach did not include traditional feature selection methods like Forward or Backward feature selection or dimensionality reduction techniques such as PCA, Lasso, or LDA. This decision was due to the nature of the data being predominantly textual and I believed the chosen features were substantial enough.

## Model Discussion

I built a decision tree classifier, random forest classifier, support vector classifier, and an extreme gradient boosting classifier. I started with the decision tree classifier as a baseline model due to its interpretability and ease of use. I then thought a random forest classifier would be the next logical step. It is an ensemble method that builds multiple decision trees and merges their results to improve the overall prediction accuracy and control over-fitting. Next, I tried an SVC model since it aims to find the hyperplane that has the maximum margin between the classes, which can help in distinctly classifying the reviews as fake or real. Finally, I used an XGBoost model. This model uses the gradient boosting framework. What it does is train a collection of simple decision trees in sequence, where each subsequent tree is built based on the prediction errors made by the preceding trees. The final prediction is a weighted sum of all of the tree predictions. This method allows the model to improve where it is not performing well. For the training and testing, I split the data with 20% of it being allocated to the test set and the remaining 80% was used for training the models.

### Decision Tree Model
After training the decision tree model and running it against the test data these were the results 

![Alt text](DT_matrix_pretune.png?raw=true "Optional Title")

And the metrics were

![Alt text](DT_metrics_pretune.png?raw=true "Optional Title")

The model performs decently in terms of accuracy but has quite poor precision and recall. The decision tree could be overfitting to the training data. The number of FPs and FNs also suggests some room for improvement particularly in reducing the misclassification of real reviews as fake. This could be due to several factors including the complexity of the decision boundaries or insufficient depth or pruning in the decision tree.

### Random Forest Model

After training the random forest and running it against the test data these were the results 

![Alt text](RF_matrix_pretune.png?raw=true "Optional Title")

And the metrics were

![Alt text](RF_metrics_pretune.png?raw=true "Optional Title")

After using the random forest algorithm the results showed a slight improvement in correctly identifying real and fake reviews over the decision tree classifier. The increase in true positives (TP) and true negatives (TN) staying about the same suggests that the random forest model is better at picking up the complex patterns in the data. The random forest's slight improvement over the decision tree is due to its ensemble nature. It aggregates the results of many decision trees and reduces the risk of overfitting. However, the improvement is not drastic as I can see in the statistics.

### Support Vector Classifier 

After training the SVC and running it against the test data these were the results 

![Alt text](SVC_matrix_pretune.png?raw=true "Optional Title")

And the metrics were

![Alt text](SVC_metrics_pretune.png?raw=true "Optional Title")

As the metrics show the model has a high recall suggesting it is quite good at flagging fake reviews. However, the precision metric is quite low which means the model tends to incorrectly label genuine reviews as fake. These recall and precision rates shows that while the SVC is efficient at identifying a high volume of fake reviews, it does so at the expense of incorrectly categorizing some real reviews. This could be a result of the model's high sensitivity to the minority class

### XGBoost Classifier

After training the XGBoost and running it against the test data these were the results 

![Alt text](XG_matrix.png?raw=true "Optional Title")

And the metrics were

![Alt text](XG_metrics.png?raw=true "Optional Title")

For the XGBoost I also used Optuna. Optuna is an open-source hyperparameter optimization framework that automates the process of finding the most effective hyperparameters. It systematically searches through various combinations of hyperparameters to determine the best set that improves the performance. After running the model with Optuna and got the metrics above. While the XGboost has a lower recall than the SVC model it outperforms it in the other statistics. It also outperforms the decision tree classifier and the random forest classifier in all the metrics. It has a higher accuracy rate which points to an overall better generalization capability. The number of true positives and true negatives suggests that the model can effectively discern between real and fake reviews, while the false positives and false negatives show a decrease compared to the SVC model. This means it has a more balanced classification with fewer instances of real reviews being mislabeled as fake.


### Conclusion


I employed a diverse array of classifiers, including the Random Forest Classifier, Decision Tree Classifier, Support Vector Classifier (SVC), and XGBoost. Although the SVC had the highest recall, observing the confusion matrix shows a subpar performance in classification. Instances of real reviews being misclassified as fake far exceeded the accurate predictions of real reviews.

Subsequent evaluation of the remaining models showcased relatively comparable confusion matrices. The Decision Tree Classifier demonstrated solid performance, but the Random Forest and XGBoost outperformed it in all metrics. Specifically, The Random Forest achieved a 76% accuracy rate, 33% precision, 36% recall, and a 60% ROC-AUC. Meanwhile, the XGBoost model excelled with a 77% accuracy, 37% precision, 46% recall, and 65% ROC-AUC. Below is a chart comparing the f1 scores of each classifier.

![Alt text](fscore_comp.png?raw=true "Optional Title")

It is evident that XGBoost is the frontrunner among the models. While the results showcase proficiency, there remains room for improvement, especially considering that other research papers have achieved accuracy rates surpassing the 80% threshold. 

## References



- Crawford, M., Khoshgoftaar, T. M., Prusa, J. D., Richter, A. N., & Al Najada, H. (2015). Survey of Review Spam Detection Using Machine Learning Techniques. [Link](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-015-0029-9)

- Alsubari, S. N., Deshmukh, S. N., Alqarni, A. A., Alsharif, N., Aldhyani, T. H. H., Alsaade, F. W., & Khalaf, O. I. (2021). Data Analytics for the Identification of Fake Reviews Using Supervised Learning. [Link](https://cdn.techscience.cn/ueditor/files/cmc/TSP_CMC_70-2/TSP_CMC_19625/TSP_CMC_19625.pdf)

- Elshrif Elmurngi, Abdelouahed Gherbi. (2017). Detecting Fake Reviews through Sentiment Analysis Using Machine Learning Techniques. [Link](https://d1wqtxts1xzle7.cloudfront.net/82899283/download_full-libre.pdf?1648599681=&response-content-disposition=inline%3B+filename%3DThe_Sixth_International_Conference_on_Da.pdf&Expires=1696619338&Signature=SkVnELulYrlDedwHLvKZ9sVORGPiCU~g4Wt3jqAdCU2Nq1i9Lt-zNd-QavY8BP9elLjJv8Yu5Z0neL27n4acWnJp34NLS~-GMXRp15XA0bOYp~QjdHzlG8zLeAD10kLLjpwRGcpcTAfwg1NdRULP8IcSv-5Wz0aDBgijQ9Cpym1KC-JOGLf0VCQ8q5lfRKS0IgIBILIS8E~v37w8egb1xk6ohyTkCJEbtx3c5RdQCeYB965xEwjmU6pZIh8JN2PlEKHMXApwKPASnIXZoPRNJGYSs5jULJQgw4Ngz3UCAJh5r2G8fqeNRXtknrmbteSLIAd9AOMdRkR~qN0mCpfpWA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA#page=74)

- Time. (2021). McCluskey, M. (2022). Inside the War on Fake Consumer Reviews. [Link](https://time.com/6192933/fake-reviews-regulation/)

- Byrne, S. (2022). Why Ratings and Reviews Are Important for Your Business. [Link](https://www.bazaarvoice.com/blog/why-ratings-and-reviews-are-important-for-your-business/#:~:text=According%20to%20consumer%20research%20we,in%20the%20consumer%20buying%20process)

- Choi, Wonil, et al. “Fake review identification and utility evaluation model using machine learning.” Frontiers in Artificial Intelligence, vol. 5, 2023. [Link](https://www.sciencedirect.com/science/article/pii/S0969698921003374?via%3Dihub#fig3)

- He, Sherry, et al. “The market for fake reviews.” Marketing Science, vol. 41, no. 5, 2022, pp. 896–921. [Link](https://pubsonline.informs.org/doi/abs/10.1287/mksc.2022.1353?casa_token=jfFrBs_Y8PoAAAAA:WmkBe2iyVjJHHNF1H7K7Nz6hn-jKHwBnIBhI_9hsYLB8O3bJSx_-0DOHcODBolNoGHs3MjBKyAfQ)
