-- Program 3 - User Rating Count by Movie
ratings = LOAD 'dataset3.csv' USING PigStorage(',') AS (movieId:int, title:chararray, userId:int, rating:float, genreId:int);
grouped_by_movie = GROUP ratings BY movieId;
user_rating_count_by_movie = FOREACH grouped_by_movie {
    users = ratings.userId;
    GENERATE group AS movieId, users, COUNT(users) AS numRatings;
};
DUMP user_rating_count_by_movie;
