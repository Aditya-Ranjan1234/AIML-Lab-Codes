-- Program 3 - Rating Statistics
ratings = LOAD 'dataset3.csv' USING PigStorage(',') AS (movieId:int, title:chararray, userId:int, rating:float, genreId:int);
grouped_data = GROUP ratings ALL;
ratings_stats = FOREACH grouped_data GENERATE MAX(ratings.rating) AS maxRating, MIN(ratings.rating) AS minRating, AVG(ratings.rating) AS avgRating;
DUMP ratings_stats;
