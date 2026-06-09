-- Program 4b - Max Temperature
data = LOAD 'prg124.csv' USING PigStorage(',') AS (CityName:chararray, Location:chararray, Year:int, Temperature:float, State:chararray, GasPresent:chararray, QuantityofGas:float);
filtered_data = FILTER data BY Temperature IS NOT NULL;
grouped_data = GROUP filtered_data ALL;
max_temperature = FOREACH grouped_data GENERATE MAX(filtered_data.Temperature) AS MaxTemp;
DUMP max_temperature;
