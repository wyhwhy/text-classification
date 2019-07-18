# text-classification
I do this project on my own when I had an internship in the Xi’an Webber Software Co., Ltd. and I analyze text-classification problem from three different aspects. First, data preprocessing. Second, traditional machine learning algorithm. Third, neural network algorithm. The data comes from Dalian University of Technology website and Xi’an Webber Software Co., Ltd. has the copyright. There are more than 10000 pieces of information including label and content of news after duplication.
## Part 1. Data preprocessing
The original method to process data is just to use jieba (a Chinese word segmentation tool in python). Apparently it is not appropriate because we have English text and mixed text with both Chinese and English. What's more, there is irrelevant information like 'you can download Foxit Reader if you can't open the file online……'. Meanwhile, I would like to validate whether using different methods to preprocess data will improve the accuracy of classification(using the easiest neural network model with 2 dense layers).Then I took the following steps:   
  1.wipe out irrelevant information   
  2.translate (using scrapy/API to get the translation, no matter what language the text is)   
  3.extract the Chinese words from the translation text   
  4.word segmentation.  
Consequently, I tried 6 methods and the accuracies on the test set are as follows  
  1.jieba (Chinese word segmentation) 86.6%  
  2.google translate+jieba 89.2%  
  3.google translate+pkuseg (a Chinese word segmentation tool developed by PKU, which is proved to have a better function than jieba) 87%  
  4.pkuseg+English word segmentation by blank space 87.1%  
  5.pkuseg+nltk(English word segmentation) 87.9%  
  6.pkuseg+google translate+pkuseg 86.5%  

## Part 2. Traditional machine learning algorithms
I tried 11 traditional machine learning algorithms to validate whether they are better than neural network algorithm when the data set is small. The results are as follows.  
  1.MultinomialNB 85.5%  
  2.SGDClassifier **93.2%**  
  3.LogisticRegression 90.8%  
  4.SVC 36.3%  
  5.LinearSVC **93.5%**  
  6.MLPClassifier(for comparison) **93.4%**  
  7.KNeighborsClassifier 82.9%  
  8.RandomForestClassifier 88.6%  
  9.GradientBoostingClassifier 90.1%  
  10.AdaBoostClassifier 58.6%  
  11.DecisionTreeClassifier 85.6%  
  12.rocchio 74.8%

## Part 3. Neural network algorithms
I tried LSTM, Bi-directional LSTM, GRU, CNN, CNN-LSTM(Bi-LSTM), LSTM(Bi-LSTM)-CNN. The results show that though having a longer running time, the accuracy of Bi-LSTM is higher than LSTM. What's more, LSTM-CNN model is better than CNN-LSTM because the latter one loses some sequential/order information. The best model is CNN and its accuracy reaches **92.2%**.
## Conclusion
1.Further preprocessing data will improve the accuracy of classification to some extent.  
2.Some traditional machine learning algorithms do have a better performance than neural network algorithms in some situations.

reference:  
http://konukoii.com/blog/2018/02/19/twitter-sentiment-analysis-using-combined-lstm-cnn-models/
