#!/usr/bin/env python
# coding: utf-8

# # DS/CMPSC 410 MiniProject Deliverable #1 
# 
# # Spring 2025
# ## Instructor: Prof. John Yen
# ## TAs: Peng Jin and Jingxi Zhu
# 
# ## Learning Objectives
# - Be able to identify frequent 1 ports, 2 port sets and 3 port sets (based on a threshold) that are scanned by scanners in the Darknet dataset.
# - Be able to adapt the Aprior algorithm by incorporating suitable threshold and pruning strategies.
# - Be able to improve the performance of frequent port set mining by suitable reuse of RDD, together with appropriate persist and unpersist on the reused RDD.
# - After successful execution in the local mode, modify the code for cluster mode, and final frequent 1-ports, 2-port sets, and 3-port sets using the big Darknet dataset (`Day_2020_profile.csv`).
# 
# ### Data
# - The small Darknet dataset 'sampled_profile.csv' and the large Darknet dataset `Day_2020_profile.csv` are available for download from Canvas, then upload to Roar under your MiniProj1 directory in work directory.
# - The thresdhold for frequent item (port) set is 400 in the local mode, and **30000 in the cluster mode**.
# 
# ### Items to submit:
# - Completed Jupyter Notebook (using small Darknet dataset `sampled_profile.csv`) in HTML format.
# - .py file for mining frequent 1 ports, 2 port sets. and 3 port sets in cluster mode using the big Darknet dataset `Day_2020_profile.csv`.
# - The log file containing the run time information in the CLUSTER mode.
# - one file of frequent 1-ports generated in the CLUSTER mode.
# - one file of frequent 2-port sets generated in the CLUSTER mode.
# - one file of frequent 3-port sets generated in the CLUSTER mode.
# - a screen shot (using ``ls -l`` terminal command) of the MiniProj1 directory, showing all files and directories 
# 
# ### Total points: 120 
# - Problem 1: 10 points
# - Problem 2: 15 points
# - Problem 3: 10 points
# - Problem 4: 10 points
# - Problem 5: 20 points
# - Problem 6: 10 points
# - Problem 7: 15 points
# - Problem 8: 30 points
#   
# ### Due: midnight, April 4, 2025

# In[1]:


import pyspark
import csv
import pandas as pd


# In[2]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, IntegerType, BooleanType, StringType, DecimalType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.clustering import KMeans


# In[3]:


ss = SparkSession.builder.appName("Mini Project #1 Freqent Port Sets Mining").getOrCreate()


# In[4]:


ss.sparkContext.setLogLevel("WARN")


# In[5]:


# ss.sparkContext.setCheckpointDir("~/scratch")


# # Problem 1 (10 points)
# - Complete the path below for reading "sampled_profile.csv" you downloaded from Canvas, uploaded to your Mini Project 1 folder. (5 points)
# - Fill in your Name (5 points) : Aidan Vesci

# In[6]:


scanner_schema = StructType([StructField("_c0", IntegerType(), False),                              StructField("id", IntegerType(), False ),                              StructField("numports", IntegerType(), False),                              StructField("lifetime", DecimalType(), False ),                              StructField("Bytes", IntegerType(), False ),                              StructField("Packets", IntegerType(), False),                              StructField("average_packetsize", IntegerType(), False),                              StructField("MinUniqueDests", IntegerType(), False),                             StructField("MaxUniqueDests", IntegerType(), False),                              StructField("MinUniqueDest24s", IntegerType(), False),                              StructField("MaxUniqueDest24s", IntegerType(), False),                              StructField("average_lifetime", DecimalType(), False),                              StructField("mirai", BooleanType(), True),                              StructField("zmap", BooleanType(), True),
                             StructField("masscan", BooleanType(), True),
                             StructField("country", StringType(), False), \
                             StructField("traffic_types_scanned_str", StringType(), False), \
                             StructField("ports_scanned_str", StringType(), False), \
                             StructField("host_tags_per_censys", StringType(), False), \
                             StructField("host_services_per_censys", StringType(), False) \
                           ])


# In[7]:


# In the cluster mode, change this line to
# Scanners_df = ss.read.csv("/storage/home/???/work/MiniProj1/Day_2020_profile.csv", schema = scanner_schema, header= True, inferSchema=False )
Scanners_df = ss.read.csv("/storage/home/ajv5723/work/MiniProj1/Day_2020_profile.csv", schema = scanner_schema,                           header=True, inferSchema=False)


# ## We can use printSchema() to display the schema of the DataFrame Scanners_df to see whether it was inferred correctly.

# In[8]:


# Scanners_df.printSchema()


# # Read scanners data, parse the ports_scanned_str into an array

# In[9]:


Scanners_df2=Scanners_df.withColumn("Ports_Array", split(col("ports_scanned_str"), "-") )


# In[10]:


Ports_Scanned_RDD = Scanners_df2.select("Ports_Array").rdd


# In[11]:


scanner_port_list_RDD = Ports_Scanned_RDD.map(lambda x: x["Ports_Array"])


# # Compute the total number of scanners scanning each port

# # Problem 2 (15 points) Complete the code below to 
# - (a) calculate the total number of scanners that scan each port (5 points)
# - (b) sort them in descending order of the count (the number of scanners scanning the port) using sortByKey (5 points)
# - (c) Save the results in a text file.

# In[12]:


port_list_RDD = scanner_port_list_RDD.flatMap(lambda x: x)


# In[13]:


port_1_RDD = port_list_RDD.map(lambda x: (x, 1) )
#port_1_RDD.take(5)


# In[14]:


port_count_RDD = port_1_RDD.reduceByKey(lambda x,y: x+y , 8)
# port_count_RDD.take(5)


# In[20]:


sorted_count_port_RDD = port_count_RDD.map(lambda x: (x[1], x[0])).sortByKey(ascending = False)


# In[21]:


# sorted_count_port_RDD.take(20)


# In[17]:


sorted_port_count = "/storage/home/ajv5723/work/MiniProj1/sorted_port_count_Cluster_new.txt"
sorted_count_port_RDD.saveAsTextFile(sorted_port_count)


# # Threshold for this MiniProject: 400  in local mode, 30000 in cluster mode

# # Problem 3 (10 points)
# ## Complete the code below to 
# - (a) filter for ports whose count of scanners (scanning the port) exceeds the thresdhold,
# - (b) save the filtered top ports in a file

# In[22]:


# This threshold value is 500 for the local mode.
# You need to change it to 30000 for the cluster mode.
threshold = 30000
freq_count_port_RDD= sorted_count_port_RDD.filter(lambda x: x[0] > threshold)
total_freq_port_count = freq_count_port_RDD.count()


# In[23]:


total_freq_port_count


# In[24]:


freq_count_port_RDD.saveAsTextFile("/storage/home/ajv5723/work/MiniProj1/freq_1_port_count_Cluster_new.txt")


# In[25]:


Top_Ports = freq_count_port_RDD.map(lambda x: x[1]).collect()


# In[26]:


# print(Top_Ports)


# In[27]:


Top_1_Port_count = len(Top_Ports)


# In[28]:


# print(Top_1_Port_count)


# # Finding Frequent Port Sets Being Scanned

# # Pruning Strategy: Because we do not need to consider any scanners that scan only 1 port in finding frequent 2-port sets or frequent 3-port sets, we can filter `multi_Ports_list_RDD` to remove those single port scanners.

# In[29]:


# scanner_port_list_RDD.take(5)


# In[30]:


# How many scanners are in this dataset?
# scanner_port_list_RDD.count()


# ## Filter 1: Because scanners who scan only one port is not needed for finding frequent 2-port sets, we can filter them out using the Python function `len`.

# # Problem 4 (10 points) 
# - (a) Complete the code below to filter for scanners that scan more than one port. (5 points)
# - (b) Compute the estimated percentage of scanners, based on the sampled data, that scan more than 1 port. (5 points)

# In[31]:


MPscanner_port_list_RDD = scanner_port_list_RDD.filter(lambda x: len(x) >= 2 )


# In[32]:


# MPscanner_port_list_RDD.take(5)


# In[33]:


multi_port_scanner_count = MPscanner_port_list_RDD.count()
# print(multi_port_scanner_count)


# In[34]:


scanner_count= scanner_port_list_RDD.count()
# print(scanner_count)


# In[35]:


pct = multi_port_scanner_count/scanner_count
pct


# # Answer to Problem 4 (b):
# The estimtated percentage of scanners that scan more than 1 port is : 32.44%

# # We will use `MPscanner_port_list_RDD` in the reamining code for finding frequent 2-port sets and frequent 3 port sets.

# ## The following two code cells demonstrate how we use Python `in` test for list to filter for scanners who scan one or more specific ports, then count the number of scanners that satisfy that criteria.

# In[36]:


count_80_23 = MPscanner_port_list_RDD.filter(lambda x: ('80' in x) and ('23' in x)).count()


# In[37]:


# print(count_80_23)


# In[38]:


count2_80_23 = MPscanner_port_list_RDD.filter(lambda x: ('80' in x)).filter(lambda x: ('23' in x)).count()


# In[39]:


# print(count2_80_23)


# # Since we will be using `MPscanner_port_list_RDD` in the reamining code for finding frequent 2-port sets and frequent 3 port sets, we display the content of a few RDD to double check that we do not see any 1-port scanners in the RDD.

# In[40]:


# MPscanner_port_list_RDD.take(5)


# # Frequent 1 Port Sets
# Earlier, we have saved the list of frequent 1 port set (the set of ports who have been scanned by more than x scanners, where x is the threshold) in the variable Top_Ports

# In[41]:


# print(Top_Ports)


# # Finding Frequent 2-Port Sets 

# ## As mentioned earlier, to check whether a scanner scans a specific top port (e.g., ith top port), we can use python code such as `(lambda x: Top_Ports[i] in x)` to filter for scanners that scan the sepcific port.
# ## We can then iterate through all pairs of Top_Ports to (1) filter for scanners that scan both ports, and (2) count the number of scanners in the filtered RDD.
# ## Below is the algorithm for finding frequent 2 port sets
# ```
# N = Total number of frequent 1-ports
# For top port index i from 0 to N-1 do:
#     filtered_MPscanner_top_port_RDD = filter MPscanner_port_list_RDD for top port index i
#     For top port index j from i+1 to N-1 do:
#         candidate_freq_2_port_set = filter filtered_MPscanner_top_port_RDD for top port index j
#         2_port_count = candidate_freq_2_port_set.count()
#         If 2_port_count > threshold:
#             Save [ [Top_port[i], Top_port[j]] , 2_port_count ] in a Pandas dataframe for frequent 2 port set
# ```

# In[42]:


Top_1_Port_count = len(Top_Ports)


# In[43]:


# print(Top_1_Port_count)


# # Adding Persist and Unpersist
# - In general, when a loop uses an RDD created outside the loop, persisting the RDD improves the efficiency because it does not need to be re-computed every iteration.
# - For example, the RDD ``MPscanner_port_list_RDD`` is used inside the first for loop in both of the algorithms above.  Therefore, it is desirable to add `MPscanner_port_list_RDD.persist()` OUTSIDE of the loop, before the loop starts.
# - In general, when an RDD is created inside a loop, and is subsequently used in another nested loop, it is desirable to apply both persist and unpersist to the RDD as follows:
# -- Add persist to the RDD before the nested loop so that it does not need to be recomputed.
# -- Add unpersist to the RDD after the nested loop so that the resources (memory, disk) used to store the RDD can be releasted.
# --- For example, the RDD ``filtered_scanners_TP_i`` is used inside the nested loop `For top port index j ...`, therefore, it needs to be persisted before entering the nested loop.  
# --- Also, the RDD ``filtered_scanners_TP_i`` is not needed once we exit the loop, therefore, it is desirable to add  `filtered_scanners_TP_i.unpersist()` at the end of the loop `For top port index i ...`

# # Problem 5 (20 points)  Complete the following code (including suitabler persist and unpersist) to find BOTH frequent 2 port sets AND frequent 3 port sets 
# - Hints:
# -- Use index `i` and `j` as looping variables to iterate through `Top_Ports` list, similar to the the way they are used in the algorithms above.
# -- Frequent two port sets are saved in Pandas dataframe `Two_Port_Sets_df` 
# -- Use two `index2` variables to save in the Pandas dataframe `Two_Port_Sets_df`.

# In[44]:


# Initialize a Pandas DataFrame to store frequent port sets and their counts 
Two_Port_Sets_df = pd.DataFrame( columns = ['Port Sets', 'count'])
# Initialize the index to Two_Port_Sets_df
index2 = 0
# Set the threshold for Frequent Port Sets to be 400 in local mode.
# This threshold needs to be changed to 10000 in the cluster mode.
threshold = 30000
MPscanner_port_list_RDD.persist()
for i in range(0, Top_1_Port_count):
    filtered_scanners_TP_i = MPscanner_port_list_RDD.filter(lambda x: Top_Ports[i] in x)
    filtered_scanners_TP_i.persist()  
    # We do not need to filter for threshold for 1-port sets because all ports in Top_Ports have a
    # frequency higher than the threshold.
    for j in range(i+1, Top_1_Port_count):
        filtered_scanners_TP_i_j = filtered_scanners_TP_i.filter(lambda x: Top_Ports[j] in x)
        port_i_j_count = filtered_scanners_TP_i_j.count()
        if port_i_j_count > threshold:
            Two_Port_Sets_df.loc[index2] = [ Top_Ports[i]+"-"+Top_Ports[j], port_i_j_count] 
            index2 = index2 + 1
            # The print statement is for running in the local mode.  It can be commented out for running in the cluster mode.
            print("Two Ports:", Top_Ports[i], " , ", Top_Ports[j], ", Count: ", port_i_j_count)
    filtered_scanners_TP_i.unpersist()


# # Create a PySpark DataFrame using the Pandas dataframes of frequent 2-port sets, then write the PySpark DataFrame (with header information)

# # Problem 6 (10 points)
# Complete the following code to save your frequent 2 port sets and 3 port sets in an output file named as ``2Ports_<your PSUID>_local.csv`` 

# In[45]:


DF2port = ss.createDataFrame(Two_Port_Sets_df)


# In[46]:


# These output file names need to be changed in the cluster mode, so that you can compare them with those from the local mode.
output_path_2_port = "/storage/home/ajv5723/work/MiniProj1/2PS_ajv5723_Cluster_new.csv"
DF2port.write.option("header", True).csv(output_path_2_port)


# # Part D Finding Frequent 3-port sets

# # Approach 1:
# ## One way To find frequent 3-port sets is to add another nested loop, inside the two loops above, to iterate three all possible frequent 3 port sets.
# ```
# N = Total number of frequent 1-ports
# For top port index i from 0 to N-1 do:
#     filtered_MPscanner_Top_port_i = filter MPscanner_port_list_RDD for top port index i
#     For top port index j from i+1 to N-1 do:
#         filtered_MPscanner_Top_port_i_j = filter filtered_MPscanner_Top_port_i for top port index j
#         port_i_j_count = filtered_MPscanner_Top_port_i_j.count()
#         If port_i_j_count > threshold:
#             Save [ [Top_port[i], Top_port[j]] , port_i_j_count ] in a Pandas dataframe for frequent 2 port set
#             For top port index k from j+1 to N-1 do:
#                 filtered_MPscanner_Top_port_i_j_k = filter filtered_MPscanner_Top_port_i_j for top port index k 
#                 port_i_j_k_count = filtered_MPscanner_Top_port_i_j_k.count()
#                 If port_i_j_k_count > threshold:
#                 Save [ [Top_port[i], Top_port[j], Top_port[k]], port_i_j_k_count ] in a Panda dataframe for frequent 3 port set
# ```

# # A More Scalable Approach:
# ## Due to the big size of the data, finding frequent 3 port set as the 2nd nested loop inside the loop for finding frequent 2 port sets is costly because it needs to maintain persisting on two RDDs needed for the outer loop.  In addition, it needs to persist and unpersist scanners for a 2 port set that exceeds the threshold so that we can iterate through possible 3rd ports for finding frequent 3 port sets.
# ## An Alternative Approach is to find frequent 3 port sets AFTER we have found frequent 2-port sets so that we can reduce the number of RDDs that need to persist at the same time.
# ## Also, we can reduce the size of scanners to consider, because we can filter out scanners that scan less than 3 ports.
# ## Below is an algorithm:
# ```
# Read scanners data, parse the ports_scanned_str into an array
# Generate an RDD containinging the list of ports scanned by each scanner.
# Top_ports = A list of ports whose scanner count is > threshold
# candidate_3PS_scanners = filter scanners for those that scan at least 3 ports
# frequent_2PS_RDD = Reads from the file created from frequent 2 port set mining
# frequent_2PS_RDD.persisit()
# for each 2PS in frequent_2PS_RDD do:
#     scanners_2PS = filter candidate_3PS_scanners for those that scan the two port set 2PS
#     if the number of scanners in scanners_2PS > threshold:
#         scanners_2PS.persist()
#         index_i = index of first port in 2PS
#         index_j = index of second port in 2PS
#         for index_k from max{index_i, index_j} +1 to len(Top_ports) do:
#             scanners_3PS = filter scanners_2PS for Top_ports[index_k]
#             if the number of scanners in scanners_3PS > threshold:
#                 Record Top_ports[index_i], Top_ports[index_j], and Top_ports[index_k] as a frequent 3PortSet together with its count
#         scanners_2PS.unpersisit()
# frequent_2PS_RDD.unpersisit()              
#         
# ```

# In[47]:


# If read from file, change this line to read from your cluster output
# DF2port = ss.read.csv("/storage/home/???/work/MiniProj1/2PS_???_local.csv", header=True, inferSchema=True)


# In[48]:


DF2port_A = DF2port.withColumn("ports array", split("Port Sets", "-") )


# In[49]:


# DF2port_A.show(3)


# In[50]:


DF2port_RDD = DF2port_A.select("ports array").rdd


# In[51]:


# DF2port_RDD.take(3)


# In[52]:


TwoPort_list = DF2port_RDD.map(lambda x: x["ports array"]).collect()


# In[53]:


# print(TwoPort_list)


# ## Filter Scanners for those that scan at least three ports

# In[54]:


Candidate_3PS_scanners = MPscanner_port_list_RDD.filter(lambda x: len(x) >= 3)


# In[55]:


Candidate_3PS_scanners.persist()


# In[56]:


MPscanner_port_list_RDD.unpersist()


# # Problem 7 (15 points) 
# ## Complete the missing code (including persist and unpersist) below for mining frequent 3 port sets
# ## and write the results (three port sets and their counts) using PySpark DataFrame.

# In[57]:


# Initialize a Pandas DataFrame to store frequent port sets and their counts 
Three_Port_Sets_df = pd.DataFrame( columns= ['Port Sets', 'count'])
# Initialize the index to Three_Port_Sets_df
index3 = 0
# Set the threshold for Frequent Port Sets to be 400 in local mode.
# This threshold needs to be changed to 30000 in the cluster mode.
threshold = 30000
Top_1_Port_count = len(Top_Ports)
for TwoPS in TwoPort_list:
    index_i = Top_Ports.index( TwoPS[0] )
    index_j = Top_Ports.index( TwoPS[1] )
    filtered_scanners_i_j = Candidate_3PS_scanners.filter(lambda x: Top_Ports[index_i] in x).filter(lambda y: Top_Ports[index_j] in y)
    filtered_scanners_i_j.persist()  
    for k in range(max(index_i, index_j)+1, Top_1_Port_count):
        filtered_scanners_i_j_k = filtered_scanners_i_j.filter(lambda x: Top_Ports[k] in x)
        port_i_j_k_count = filtered_scanners_i_j_k.count()
        if port_i_j_k_count > threshold:
            Three_Port_Sets_df.loc[index3] = [ Top_Ports[index_i]+"-"+Top_Ports[index_j]+"-"+Top_Ports[k], port_i_j_k_count] 
            index3 = index3 + 1
            # The print statement is for running in the local mode.  It can be commented out for running in the cluster mode.
            print("Three Ports:", Top_Ports[index_i], " , ", Top_Ports[index_j], " , ", Top_Ports[k], ", Count: ", port_i_j_k_count)
    filtered_scanners_i_j.unpersist()


# In[58]:


DF3port = ss.createDataFrame(Three_Port_Sets_df)


# In[59]:


# These output file names need to be changed in the cluster mode, so that you can compare them with those from the local mode.
output_path_3_port = "/storage/home/ajv5723/work/MiniProj1/3PS_ajv5723_Cluster_new.csv"
DF3port.write.option("header", True).csv(output_path_3_port)


# In[60]:


ss.stop()


# # Part E (cluster mode): Finding frequent 2-port sets and 3-port sets from the large dataset.

# # Problem 8 (30 points)
# - Remove .master("local") from SparkSession statement
# - Change the input file to "Day_2020_profile.csv"
# - Change the threshold from 400 to 30000.
# - Change the output files to two different directories from the ones you used in local mode.
# - Export the notebook as a .py file
# - Run pbs-spark-submit on ICDS Roar 
# - Submit the following items:
# -- (a) the .py file for cluster mode (5%)
# -- (b) the log file containing the run time information for the cluster mode (5%)
# -- (b) One output file for frequent 2-port sets and one output file for frequent 3-port sets generated in the cluster mode. (10%)
# -- (c) A screen shot (generated using `ls -l` terminal command) of your `MiniProj1` that shows all files and directories. (5%)
# -- (d) Discuss (in the cell below) three things you noticed that are interesting/surprising from the frequent 3-port sets (5%)

# # Your Answer to Exercise 8 (d):
# 1) Ports 17132 and 17140 consistently appear, often with a third varying port. This suggests these two ports are central to the communication or traffic being analyzed.
# 2) Ports with 17132 and another 17*** ranged port have way higher counts then a lot of other trios, showing that these ports might dominate the traffic flows.
# 3) The higher port numbers specifically in the range of 30000+, have very low counts suggesting that they are maybe not used as much as other ports.

# In[ ]:




