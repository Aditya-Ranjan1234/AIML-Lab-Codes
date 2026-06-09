-- Program 5 - Query 1: Select with Recommendation Integer
SELECT USER_ID, MOVIE_ID, ACTIVITY,
    CASE
        WHEN RECOMMENDED = 'Y' THEN 1
        ELSE 0
    END AS RECOMMEND_INT
FROM movie_log
WHERE ACTIVITY IN (4,5,2,1)
AND GENRE_ID IS NOT NULL
LIMIT 25;
