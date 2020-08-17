

#import the libraries
import pandas as pd  #pandas is used for reading the data set
import numpy as np   #numpy is used for creating multi dimensional array
#Read the file which contains information like user id,movie id,rating and timestamp.Here user id is information of unique users who are giving some ratings to movies of some specific movie id)
ratings=pd.read_csv("ratings.csv")
ratings.head()
#Another dataset contains the mapping information of movie_id and title of the movie
#Check all the movies and their respective ids
movie_titles=pd.read_csv("titles.csv")
movie_titles.head()
#Merge two dataframe to create single data frame by using merge method from dataframe
ratings_movie=pd.merge(ratings,movie_titles,on='movie_id').drop(['timestamp'],axis=1)
ratings_movie.head()

#EXPLORATORY DATA ANALYSIS
#import necessary libraries
import matplotlib.pyplot as plt #matplotlib and seaborn is used for visualization purpose
import seaborn as sbs
sbs.set_style('white')
%matplotlib inline

#Create a ratings dataframe with average rating and number of ratings.
ratings_movie.groupby('title')['rating'].mean().sort_values(ascending=False).head() #ascending=False means data will be sorted in descending order
#create a dataframe for how many numbers of rating are given with respect to the movies
ratings_movie.groupby('title')['rating'].count().sort_values(ascending=False).head()
ratings_df=pd.DataFrame(ratings_movie.groupby('title')['rating'].mean())
ratings_df.head()
#set the number of ratings column
ratings_df['number of ratings']=pd.DataFrame(ratings_movie.groupby('title')['rating'].count())
ratings_df.head()
plt.figure(figsize=(10,4))
ratings_df['number of ratings'].hist(bins=70) #number of ratings are distributed as seen in graph below
#creating histogram with respect to ratings
plt.figure(figsize=(10,4))
ratings_df['rating'].hist(bins=70)

#Recommending similar movies
user_ratings=ratings_movie.pivot_table(index=['user_id'],columns=['title'],values='rating')
user_ratings.head()
ratings_df.sort_values('number of ratings',ascending=False).head(15)
#Drop few movies from dataframe which have less than 5 users who rated it .
user_rate=user_ratings.dropna(thresh=5,axis=1).fillna(0) #All nan values replaced by zeros
user_rate.head()

#Consider any three movies.For Ex:Aakrosh ,3 Idiots and LunchBox
ratings_df.head()
#Take the user ratings for these movies
aakrosh_user_rating=user_ratings['Aakrosh (2010)']
three_idiots_user_rating=user_ratings['3 Idiots (2009)']
lb_user_rating=user_ratings['The Lunchbox (2013)']
aakrosh_user_rating.head()
#Similarity matrix.To get correlations between two panda series use corrwith() method.
similarity=user_rate.corrwith(aakrosh_user_rating)
similar_to_three_idiots=user_rate.corrwith(three_idiots_user_rating)
similar_to_lb=user_rate.corrwith(lb_user_rating)
corr_aakrosh_movie=pd.DataFrame(similarity,columns=['Correlation'])
#Remove NAN values 
corr_aakrosh_movie.dropna(inplace=True)
corr_aakrosh_movie.head()
corr_aakrosh_movie.sort_values('Correlation',ascending=False).head(10)
corr_aakrosh_movie = corr_aakrosh_movie.join(ratings_df['number of ratings'])
corr_aakrosh_movie[corr_aakrosh_movie['number of ratings']>100].sort_values('Correlation',ascending=False).head() #sort the values if the number of ratings is greater than 100
corr_aakrosh_movie.head()
corr_three_idiots = pd.DataFrame(similar_to_three_idiots,columns=['Correlation'])
corr_three_idiots.dropna(inplace=True)
corr_three_idiots = corr_three_idiots.join(ratings_df['number of ratings'])
corr_three_idiots[corr_three_idiots['number of ratings']>100].sort_values('Correlation',ascending=False).head()
corr_lb = pd.DataFrame(similar_to_lb,columns=['Correlation'])
corr_lb.dropna(inplace=True)
corr_lb = corr_lb.join(ratings_df['number of ratings'])
corr_lb[corr_lb['number of ratings']>100].sort_values('Correlation',ascending=False).head()

