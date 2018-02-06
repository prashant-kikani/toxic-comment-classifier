# toxic-comment-classifier

This project is to classify toxic and abusive comments from huge bunch of text.<br />
I have gave training for these 6 type of toxic comments : <br/>
1. toxic<br/>
2. severe_toxic<br/>
3. obscene<br/>
4. threat<br/>
5. insult<br/>
6. identity_hate<br/>
<br/>
It predicts <b>probability of toxicity</b> for above defined classes. <br/>
A comment may be classified in one or more classes. As for example a comment may be "insulting" to someone but may not be "threatning".<br/><br/>
Download train/test data from here : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data <br/>
<br/>

Data cleaning is also done before this training and testing. I only have kept alphabetic latters and some punctuation marks. No numbers or any other sysmbols. <br/>  

We can do this task on any given chunk of data. We can pick the highest probability class to choose the type of toxicity.<br/>

Have a look at https://github.com/prashant-kikani/toxic-comment-classifier/blob/master/test.ipynb to get the clear idea.<br/>
