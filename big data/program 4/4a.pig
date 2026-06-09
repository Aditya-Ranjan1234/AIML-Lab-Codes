-- Program 4a - Group by Year with Data Bag
data = LOAD 'prg124.csv' USING PigStorage(',') AS (CityName:chararray, Location:chararray, Year:int, Temperature:float, State:chararray, GasPresent:chararray, QuantityofGas:float);
grouped_by_year = GROUP data BY Year;
year_bag = FOREACH grouped_by_year GENERATE group AS Year, TOBAG(data) AS DataBag;
DUMP year_bag;
