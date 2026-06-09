-- Program 3 - Distinct Users
ratings = LOAD 'dataset3.csv' USING PigStorage(',') AS (movieId:int, title:chararray, userId:int, rating:float, genreId:int);
valid_ratings = FILTER ratings BY userId IS NOT NULL;
distinct_users = DISTINCT valid_ratings;
distinct_userid = FOREACH distinct_users GENERATE userId;
DUMP distinct_userid;
