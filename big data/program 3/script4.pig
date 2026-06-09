-- Program 3 - High Rating Movies Count
ratings = LOAD 'dataset3.csv' USING PigStorage(',') AS (movieId:int, title:chararray, userId:int, rating:float, genreId:int);
high_rating = FILTER ratings BY rating > 3;
grouped_by_movie = GROUP high_rating BY movieId;
movie_count = FOREACH grouped_by_movie GENERATE group AS movieId;
distinct_movie_count = GROUP movie_count ALL;
final_count = FOREACH distinct_movie_count GENERATE COUNT(movie_count) AS numHIGHRATINGMOVIES;
DUMP final_count;
