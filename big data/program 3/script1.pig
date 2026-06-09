-- Program 3 - Movie Rating Count
ratings = LOAD 'dataset3.csv' USING PigStorage(',') AS (movieId:int, title:chararray, userId:int, rating:float, genreId:int);
grouped_ratings = GROUP ratings BY movieId;
movie_rating_count = FOREACH grouped_ratings GENERATE group AS movieId, COUNT(ratings) AS numRatings;
DUMP movie_rating_count;
