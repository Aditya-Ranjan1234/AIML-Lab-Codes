-- Program 4c - Average Temperature by State (2010-2012)
data = LOAD 'prg124.csv' USING PigStorage(',') AS (CityName:chararray, Location:chararray, Year:int, Temperature:float, State:chararray, GasPresent:chararray, QuantityofGas:float);
filtered_data = FILTER data BY (Year >= 2010 AND Year <= 2012);
valid_temperature_data = FILTER filtered_data BY Temperature IS NOT NULL;
grouped_by_state = GROUP valid_temperature_data BY State;
avg_temperature = FOREACH grouped_by_state GENERATE group AS State, AVG(valid_temperature_data.Temperature) AS AvgTemp;
STORE avg_temperature INTO '/outputp4c' USING PigStorage(',');
