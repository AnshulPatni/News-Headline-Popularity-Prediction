******GENERAL INSTRUCTIONS AND INFORMATIONS FOR WEBAPPLICATION CODE******

The source code includes 'app.py'(can be executed from terminal) file inside 'flask-app'. The required data are kept inside 'dataset' folder inside 'flask-app' folder which includes following files - 
1. final_news.csv
2. Facebook_Economy.csv
3. Facebook_Microsoft.csv
4. Facebook_Obama.csv
5. Facebook_Palestine.csv
6. GooglePlus_Economy.csv
7. GooglePlus_Microsoft.csv
8. GooglePlus_Obama.csv
9. GooglePlus_Palestine.csv
10. LinkedIn_Economy.csv
11. LinkedIn_Microsoft.csv
12. LinkedIn_Obama.csv
13. LinkedIn_Palestine.csv
14. test_sentences.csv

All other files are intermediate files generated while executing the code.

****These files are manually pre-processed in following ways - ****

1. Removed unnecessary columns
2. Removed NaN values
___________________________________________________________________________

Check all dependencies and install requirements inside flask environment-

Install all python important library - Numpy, Scipy, Pandas, Matplotlib, sklearn, Tensorflow, Keras, nltk

___________________________________________________________________________

****To run 'app.py' file.****

$ python3 app.py

A web application will open with a box and button. Type a test sentence and click on submit button. 

___________________________________________________________________________
****Ouput****

1. Category predicted
2. Five Most similar sentences
3. Time series analysis predictions graphs
 The source code only contain LSTM output. However, we have compared three time series analysis models.
4. Future time prediction graph

