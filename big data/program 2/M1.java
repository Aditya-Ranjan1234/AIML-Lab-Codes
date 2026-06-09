import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import java.io.IOException;

public class M1 extends Mapper<LongWritable, Text, Text, IntWritable> {
    private static final int MISSING = 9999;

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] fields = line.split(",");
        if (fields.length > 1 && !fields[2].equals("Year")) {
            try {
                int temperature = Integer.parseInt(fields[3]);
                context.write(new Text(fields[2]), new IntWritable(temperature));
            } catch (NumberFormatException e) {
            }
        }
    }
}
