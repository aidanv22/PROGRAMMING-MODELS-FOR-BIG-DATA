#!/usr/bin/env python
# coding: utf-8

# # DS/CMPSC 410 Spring 2025
# # Instructor: Professor John Yen
# # TA: Peng Jin and Jingxi Zhu
# 
# # Lab 5: Data Frames, DF-based Aggregation, and Top Movie Reviews 
# 
# # The goals of this lab are for you to be able to
# ## - Use Data Frames in Spark for Processing Structured Data
# ## - Perform Basic DataFrame Transformation: Filtering Rows and Selecting Columns of DataFrame
# ## - Create New Column of DataFrame using `withColumn`
# ## - Use DF SQL Function split to transform a string into an Array
# ## - Filter on a DF Column that is an Array using `array_contains`
# ## - Use `Join` to integrate DataFrames 
# ## - Use `GroupBy`, followed by `count` and `sum` DF transformation to calculate the total count (of rows) and the summation of a DF column (e.g., reviews) for each group (e.g., movie).
# ## - Perform sorting on a DataFrame column
# ## - Apply the obove to find Movies in a Genre of your choice that has good reviews with a significant number of ratings (use 10 as the threshold for local mode, 100 as the threshold for cluster mode).
# ## - After completing all exercises in the Notebook, convert the code for processing large reviews dataset and large movies dataset to find movies with top average ranking with at least 100 reviews for a genre of your choice.
# 
# ## Total Number of Exercises: 
# - Exercise 1: 5 points
# - Exercise 2A: 5 points
# - Exercise 2B: 5 points
# - Exercise 3A: 5 points
# - Exercise 3B: 5 points
# - Exercise 4: 5 points
# - Exercise 5A: 5 points
# - Exercise 5B: 5 points
# - Exercise 6: 5 points
# - Exercise 7: 5 points
# - Exercise 8: 10 points
# - Exercise 9: 10 points
# - Part B (Exercise 10): 
# - Correct .py file for spark-submit (10 points)
# - Log file of successful pbs-spark-submit (10 points)
# - Correct output file (cluster mode) for movies, sorted by average reviews, filtered for having been reviewed by more than the mean of reviewers/movie (10 points) 
# ## Total Points: 100 points
# 
# # Due: midnight, February 16th, 2025
# 

# ## The first thing we need to do in each Jupyter Notebook running pyspark is to import pyspark first.

# In[1]:


import pyspark


# ### Once we import pyspark, we need to import "SparkContext".  Every spark program needs a SparkContext object
# ### In order to use Spark SQL on DataFrames, we also need to import SparkSession from PySpark.SQL

# In[2]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row


# ## We then create a Spark Session variable (rather than Spark Context) in order to use DataFrame. 
# - Note: We temporarily use "local" as the parameter for master in this notebook so that we can test it in ICDS Roar.  However, we need to REMOVE .master("local") before we submit it  to run in cluster mode.

# In[3]:


ss=SparkSession.builder.appName("Lab 5 Top Reviews").getOrCreate()


# In[4]:


ss.sparkContext.setLogLevel("WARN")


# # Replace the question marks in the path below with your User ID (i.e., your home directory) to specify your scratch directory.

# In[5]:


ss.sparkContext.setCheckpointDir("/storage/home/ajv5723/scratch")


# # Exercise 1 (5 points) 
# - (a) Add your name below AND 
# - (b) replace the path below in both `ss.read.csv` statements with the path of your home directory.
# 
# ## Answer for Exercise 1 (Double click this Markdown cell to fill your name below.)
# - a: Student Name: Aidan Vesci

# In[6]:


rating_schema = StructType([ StructField("UserID", IntegerType(), False ),                             StructField("MovieID", IntegerType(), True),                             StructField("Rating", FloatType(), True ),                             StructField("RatingID", IntegerType(), True ),                            ])


# In[7]:


ratings_DF = ss.read.csv("/storage/home/ajv5723/work/Lab5/ratings_large.csv", schema= rating_schema, header=False, inferSchema=False)
# In the cluster mode, we need to change the input path as well as the header parameter:  `header=False` because the large rating file does not have header.


# In[8]:


movie_schema = StructType([ StructField("MovieID", IntegerType(), False),                             StructField("MovieTitle", StringType(), True ),                             StructField("Genres", StringType(), True ),                            ])


# In[9]:


movies_DF = ss.read.csv("/storage/home/ajv5723/work/Lab5/movies_large.csv", schema=movie_schema, header=False, inferSchema=False)
# In the cluster mode, we need to change the input path as well as the header parameter: `header=False` because the large movie file does not have header.


# In[10]:


# movies_DF.printSchema()


# In[11]:


# movies_DF.show(10)


# # Transforming DataFrame to RDD
# ## Use Case: Counting movies by Generes
# Suppose we want to count the number of movies in each generes.  There are two ways to do this: 
# - (1) using RDD-based map and reduceByKey, and 
# - (2) using DF-based groupBy and aggregation operator (count).
# We will first discuss RDD-based approach.  Later in this notebook, we will introduce DF-based aggregation.
# 
# # RDD-based aggregation
# In order to use RDD-based aggregation on a DataFrame, we need to 
# - (a) Select the MovieID and Genres columns from the movies DataFrame into a new DataFrame.
# - (b) Convert the new DataFrame into an RDD.
# - (c) Use map to split the Genres (a string) of each movie using the deliminator "|" (similar to the way we splitted a tweet into a list of tokens using the deliminator " ").
# - (d) Flatted the list of genres into a giantic list of genres using `flatMap`.
# - (e) Use map and reduceByKey to count the total number that each genres occur in the giantic list (similar to the way we compute hashtags in previous labs).

# In[12]:


# step (a)
movies_genres_DF = movies_DF.select("MovieID","Genres")


# ## step (b)
# ## The method "rdd", when applied to a DataFrame, returns an RDD representation of the DataFrame.
# - Each element of the converted RDD is a `Row` object, which corresponds to each row of the DataFrame.  A column value in a row object can be accessed by the column value, as illustrated below for the column `Genres`.

# In[13]:


# step (b)
movies_genres_rdd = movies_genres_DF.rdd


# In[14]:


# movies_genres_rdd.take(3)


# In[15]:


# movies_genres_rdd.take(3)[0]['Genres']


# # Exercise 2A (5 points)
# Complete the code below to obtain the Genres of the second movie in `movies_genres_rdd`

# In[17]:


movies_genres_rdd.take(3)[1]['Genres']


# In[18]:


# step (c)
splitted_genres_rdd = movies_genres_rdd.map(lambda x: x['Genres'].split('|'))


# In[19]:


# splitted_genres_rdd.take(3)


# In[20]:


# step (d)
flattened_genres_rdd = splitted_genres_rdd.flatMap(lambda x: x)
# flattened_genres_rdd.take(10)


# # Exercise 2B (5 points)
# Complete the code below to compute the total number of movies in each genre (using map and reduceByKey in a way similar to counting hashtag in previous labs), and save the result in a subdirectory in your Lab5 directory.

# In[21]:


genre_1_rdd = flattened_genres_rdd.map(lambda x: (x, 1))


# In[22]:


genre_count_rdd = genre_1_rdd.reduceByKey(lambda x, y: x+y , 5)


# In[23]:


# genre_count_rdd.take(10)


# In[24]:


genre_count_rdd.saveAsTextFile("/storage/home/ajv5723/work/Lab5/Genre_count_cluster.txt")


# In[25]:


# ratings_DF.printSchema()


# In[26]:


# ratings_DF.show(5)


# # 2. DataFrames Transformations
# DataFrame in Spark provides higher-level transformations that are convenient for selecting rows, columns, and for creating new columns.  These transformations are part of Spark SQL.
# 
# ## 2.1 `where` DF Transformation for Filtering/Selecting Rows
# Select rows from a DataFrame (DF) that satisfy a condition.  This is similar to "WHERE" clause in SQL query language.
# - One important difference (compared to SQL) is we need to add `col( ...)` when referring to a column name. 
# - The condition inside `where` transformation can be an equality test ('=='), greater-than test ('>'), or less-then test ('<'), as illustrated below.

# # `show` DF action
# The `show` DF action is similar to `take` RDD action. It takes a number as a parameter, which is the number of elements to be randomly selected from the DF to be displayed. 

# In[27]:


movies_DF.where(col("MovieTitle")== "Jurassic Park (1993)").show()


# In[28]:


ratings_DF.where(col("Rating") > 2).show(5)


# # `count` DF action
# The `count` action returns the total number of elements in the input DataFrame.

# In[29]:


ratings_DF.filter(col("Rating") > 3).count()


# # Notice: DataFrame, like RDD, is immutable
# - Did the filter method above change the content of ratings_DF?

# In[30]:


# ratings_DF.count()


# # Exercise 3A (5 points) Filtering DF Rows
# ### Complete the following statement to (1) select the `ratings_DF` DataFrame for reviews that are above 4, and (2) count the total number of such reviews.

# In[31]:


high_review_count = ratings_DF.where(col("Rating") > 4).count()
print(high_review_count)


# ## 2.2 DataFrame Transformation for Selecting Columns
# 
# DataFrame transformation `select` is similar to the projection operation in SQL: it returns a DataFrame that contains all of the columns selected.

# In[32]:


movies_DF.select("MovieID","MovieTitle").show(5)


# In[33]:


ratings_DF.select("Rating").show(5)


# # Selecting Columns from a DF
# ## The following PySpark statement to (1) select only `MovieID` and `Rating` columns from `ratings_DF`, and (2) save it in a DataFrame called `movie_rating_DF`.

# In[34]:


movie_rating_DF = ratings_DF.select("MovieID", "Rating")


# In[35]:


# movie_rating_DF.show(5)


# # 2.3 Statistical Summary of Numerical Columns
# DataFrame provides a `describe` method that provides a summary of basic statistical information (e.g., count, mean, standard deviation, min, max) for numerical columns.

# In[36]:


# ratings_DF.describe().show()


# ## RDD has a histogram method to compute the total number of rows in each "bucket".
# The code below selects the Rating column from `ratings_DF`, converts it to an RDD, which maps to extract the rating value for each row, which is used to compute the total number of reviews in 6 buckets: 
# - 0 <= reviews < 1 
# - 1 <= reviews < 2 
# - 2 <= reviews < 3 
# - 3 <= reviews < 4 
# - 4 <= reviews < 5 
# - 5 <= reivews < 6 

# In[37]:


ratings_DF.select(col("Rating")).rdd.map(lambda row: row['Rating']).histogram([0,1,2,3,4,5,6])


# # Exercise 3B (5 points)
# Based on the result, answer the following questions:
# - (a) How many reviews are 5?
# - (b) How many reviews are below 1 (i.e., 0.5)?

# # Solution for Exercise 3B
# - (a)15095
# - (b)1101

# # 3. Transforming the Generes Column into Array of Generes 
# ## We want transform a column Generes, which represent all Generes of a movie using a string that uses "|" to connect the Generes so that we can later filter for movies of a Genere more efficiently.
# ## This transformation can be done using `split` Spark SQL function (which is different from python `split` function)
# ## Because the character '|' is a special character in Python, we need to add escape character (back slash \) in front of it.

# In[38]:


Splitted_Generes_DF= movies_DF.select(split(col("Genres"), '\|'))
# Splitted_Generes_DF.show(5)


# ## 3.1 Adding a Column to a DataFrame using withColumn
# 
# # `withColumn` DF Transformation
# 
# We often need to transform content of a column into another column. For example, it is desirable to transform the column Genres in the movies DataFrame into an `Array` of genres that each movie belongs, we can do this using the DataFrame method `withColumn`.

# ### Creates a new column called "Genres_Array", whose values are arrays of genres for each movie, obtained by splitting the column value of "Genres" for each row (movie).

# In[39]:


moviesGA_DF= movies_DF.withColumn("Genres_Array",split(col("Genres"), '\|') )


# In[40]:


# moviesGA_DF.printSchema()


# In[41]:


# moviesGA_DF.show(5)


# # array_contains (SQL function)
# - An SQL function is a function that can be used in `where` and `filter` methods of DataFrame for selecting rows.
# - `array_contains` is an SQL function that checks whether an array column (of type `array`) contains a specific value.
# - For example, we can use `arrang_contains` to filter for all movies whose `Genres_Array` contain a specific genre (e.g., 'Adventure').

# # Exercise 4 (5 points)
# Complete the code below to select, from the DataFrame `moviesGA_DF', all movies in the Adventure genre.

# In[42]:


from pyspark.sql.functions import array_contains
movies_adv_genre_DF = moviesGA_DF.filter(array_contains('Genres_Array', "Adventure"))


# In[43]:


# movies_adv_genre_DF.show(5)


# # A DF-based approach to aggregate 
# We mentioned earlier there are two ways to perform aggregation: (1) an RDD-based aggregation, and (2) a DF-based aggregation.  
# 
# While an RDD-based aggregation uses key-value pairs and reduceByKey, DF-based aggregation performs aggregation using `groupBy`, which groups a DataFrame based on the value of a specific column.  Below are some examples:
# - If a DataFrame has two columns: `MovieID` and `Rating`, applying `groupBy` on the `MovieID` column enable us to add all of each movie's reviews into a total sum, which can be used to calculate an average review (by dividing it by the total number of reviews) for each movie.

# # `groupBy` DF transformation
# Taking a column name (string) as the parameter, the transformation groups rows of the DF based on the column.  All rows with the same value for the column is grouped together.  The result of groupBy transformation is often followed by an aggregation (e.g., sum, count) across all rows in each group.  
# 
# # `sum` DF transformation
# Taking a column name (string) as the parameter. This is typically used after `groupBy` DF transformation. The transformation `sum` adds the value of the input column (which should be a number) across all rows in the each group. 
# - The DataFrame created has two column: (1) the column used in `groupBy', and (2) a column named `sum(...)` where ... is the name of the column being added.
# 
# # `count` DF transformation
# Returns the number of rows in the DataFrame.  When `count` is used after `groupBy`, it returns a DataFrame with a column called "count" that contains the total number of rows for each group generated by the `groupBy`.
# - Similar to `sum`, the DataFrame created by `count` has two columns: (1) the column used in `groupBy`, and (2) a column named `count`.

# In[44]:


Movie_RatingSum_DF = ratings_DF.groupBy("MovieID").sum("Rating")


# In[45]:


# Movie_RatingSum_DF.show(4)


# # Exercise 5A (5 points)
# Complete the code below to calculate the total number of reviews for each movies.

# In[46]:


Movie_RatingCount_DF = ratings_DF.groupBy("MovieID").count()


# In[47]:


# Movie_RatingCount_DF.show(4)


# # Exercise 5B (5 points)
# Complete the code below to calculate the average number of reviews and maximal number of reviews across all movies. 
# - Save the average number of reviews in a variable `Reviewers_mean`, which we will use later to filter for movies in a genre (only including movies that received more than the average number of reviews).

# In[48]:


from pyspark.sql.functions import avg, max


# In[49]:


Movie_RatingCount_DF.select( avg(col("count")) ).show()


# In[50]:


Reviewers_mean_rdd = Movie_RatingCount_DF.select( avg(col("count")) ).rdd


# In[51]:


# Reviewers_mean_rdd.take(1)


# In[52]:


Reviewers_mean=Reviewers_mean_rdd.take(1)[0]['avg(count)']


# In[53]:


# print(Reviewers_mean)


# # Maximal Reviewers for a Movie
# We can also find out the maximal reviews a movie received in this small dataset.

# In[54]:


Movie_RatingCount_DF.select( max(col("count")) ).show()


# In[55]:


Reviewers_max_rdd = Movie_RatingCount_DF.select( max(col("count")) ).rdd


# In[56]:


Reviewers_max = Reviewers_max_rdd.take(1)[0]['max(count)']


# # We can use histogram function to understand the distribution of reviewers/movie.  We want to make sure the boundary of the highest bucket is larger than the maximal reviewers for a movie (hence, adding 1 to `Reviewers_max`).

# In[57]:


Movie_RatingCount_DF.select("count").rdd.map(lambda x: x['count']).histogram([0, 5, 10, 15, 20, 25, 50,100, Reviewers_max +1]) 


# # 5. Join Transformation on Two DataFrames

# # Exercise 6 (5 points)
# Complete the code below to (1) perform DF-based inner join on the column MovieID, and (2) calculate the average rating for each movie.

# In[58]:


Movie_Rating_Sum_Count_DF = Movie_RatingSum_DF.join(Movie_RatingCount_DF,"MovieID", 'inner')


# In[59]:


# Movie_Rating_Sum_Count_DF.show(4)


# In[60]:


Movie_Rating_Count_Avg_DF = Movie_Rating_Sum_Count_DF.withColumn("AvgRating", col("sum(Rating)") / col("count")  )


# In[61]:


# Movie_Rating_Count_Avg_DF.show(4)


# ##  Next, we want to join the Movie_Rating_Count_Avg_DF with moviesG2_DF so that we have other information about the movie (e.g., titles, Genres_Array)

# In[62]:


joined_DF = Movie_Rating_Count_Avg_DF.join(moviesGA_DF,'MovieID', 'inner')


# In[63]:


# moviesGA_DF.printSchema()


# In[64]:


# joined_DF.printSchema()


# In[65]:


# joined_DF.show(5)


# # 6. Filter DataFrame on an Array Column of DataFrame Using `array_contains`
# 
# ## Exercise 7 (5 points)
# Complete the following code to filter for a genre of your choice.

# In[67]:


from pyspark.sql.functions import array_contains
SelectGenreAvgRating_DF = joined_DF.filter(array_contains('Genres_Array',                                                'Comedy')).select("MovieID","AvgRating","count","MovieTitle","Genres","Genres_Array")


# In[68]:


# SelectGenreAvgRating_DF.show(5)


# In[69]:


# SelectGenreAvgRating_DF.count()


# In[70]:


# SelectGenreAvgRating_DF.describe().show()


# In[71]:


SortedSelectGenreAvgRating_DF = SelectGenreAvgRating_DF.orderBy('AvgRating', ascending=False)


# In[72]:


# SortedSelectGenreAvgRating_DF.show(10)


# ## We noticed many of the movie with high average rating were rated by a small number of reviewers. Should we filter them out?  How? What threshold should we use?

# # Exercise 8 (10 points)
# Use DataFrame method `where` or `filter` to find all movies (in your choice of genre) that have received reviews more than `Reviewers_mean` calculated earlier.

# In[73]:


SortedFilteredSelectGenreAvgRating_DF = SortedSelectGenreAvgRating_DF.where(col("count") > Reviewers_mean)


# In[74]:


# SortedFilteredSelectGenreAvgRating_DF.show(5)


# # Saving a DataFrame to a CSV file
# Because a column of `array` type (like `Genres_Array`) can not be saved in a csv file, we want to select all of the other columns into a dataframe so that it is ready to be written to CSV files.

# ## Exercise 9 (10 ponts)
# Complete the code below to 
# - (1) Select all of the columns of `SortedFilteredSelectGenreAvgRating_DF`, except 'Genres_Array', to a DataFrame
# - (2) save the DataFrame as CSV files in an output directory in Lab5.

# In[75]:


Top_Movies_Your_Genre_DF = SortedFilteredSelectGenreAvgRating_DF.select("MovieID","AvgRating","count","MovieTitle","Genres",)


# In[76]:


# Top_Movies_Your_Genre_DF.show(5)


# # Writing the column names of a DataFrame as header in the output CSV file
# - The header parameter in `options` of DataFrame `write` method allows us to save the column names of the DataFrame as the header in the output csv.
# - We just specify the value of the header parameter to be `True`.

# In[78]:


output_path = "/storage/home/ajv5723/work/Lab5/TopMovies4YourGenre_cluster_1"
Top_Movies_Your_Genre_DF.write.options(header=True).csv(output_path)


# In[79]:


ss.stop()


# # Exercise 10 (30 points)
# Enter the following information based on the results of running your code (.py file) on large datasets in the cluster. The correct file submission is required for each question.
# - (a) Submit your .py file for cluster mode and answer question 1 below (10 points)
# - (b) Submit your log file for a successful execution in the cluster mode and answer question 3 below (10 points)
# - (c) Submit the first file in your output directory for movies in your choice of genre, sorted by average rating, that receive more reviews than `Reviewers_mean` in the cluster mode, and answer question 2 below. (10 points)

# 1. What is your choice of the genre for your analysis? 
# 2. What are the top five movies in the genre?
# 3. What is the computation time your job took? 

# # Answer to questions for Exercise 10
# - 1.
# - 2. 
# - 3.
