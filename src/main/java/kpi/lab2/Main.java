package kpi.lab2;

import org.apache.commons.math3.distribution.TDistribution;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;

import static org.apache.spark.sql.functions.col;

public class Main {

    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder()
                .appName("myApp")
                .master("local[*]")
                .getOrCreate();

        try (JavaSparkContext sparkContext = new JavaSparkContext(sparkSession.sparkContext())) {
            Dataset<Row> data = sparkSession.read()
                    .option("header", true)
                    .csv("Training_set_ccpp.csv")
                    .limit(100);
            Dataset<Row> convertedData = data.select(
                    col("AT").cast("double"),
                    col("EV").cast("double"),
                    col("AP").cast("double"),
                    col("RH").cast("double"),
                    col("PE").cast("double")
            );
            System.out.println("Correlation between AT and PE: " + convertedData.stat().corr("PE", "AT"));

            String[] headers = {"AT"};
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(headers)
                    .setOutputCol("features");

            Dataset<Row> assembledData = assembler
                    .transform(convertedData)
                    .select("features", "PE");

            Dataset<Row>[] splits = assembledData.randomSplit(new double[]{0.8, 0.2});
            Dataset<Row> trainingData = splits[0];
            Dataset<Row> testData = splits[1];

            LinearRegression lr = new LinearRegression().setLabelCol("PE").setFeaturesCol("features");

            LinearRegressionModel model = lr.fit(trainingData);
            System.out.println("R^2 value: " + model.summary().r2());
            System.out.println("pValues: " + Arrays.toString(model.summary().pValues()));

            System.out.println("Coefficients: ");
            for (int i = 0; i < headers.length; i++) {
                System.out.println(headers[i] + ": " + model.coefficients().toArray()[i]);
            }

            // Print the coefficients and intercept
            System.out.println("Coefficients: " + model.coefficients() + " Intercept: " + model.intercept());


            Dataset<Row> predictions = model.transform(testData);
            double confidenceLevel = 0.95; // Рівень довіри
            double standardError = model.summary().coefficientStandardErrors()[0]; // Стандартна похибка прогнозу
            System.out.println("Standard error: " +standardError);
            System.out.println(Arrays.toString(model.summary().coefficientStandardErrors()));
            long numDataPoints = data.count(); // Кількість спостережень
            double criticalValue = new TDistribution(numDataPoints).inverseCumulativeProbability(1.0 - (1.0 - confidenceLevel) / 2); // Критичне значення для t-розподілу з n-2 ступенями свободи
            System.out.println("Critical value: " + criticalValue);

            System.out.println(model.coefficients().apply(0));
            double lowerBound = model.coefficients().apply(0) - criticalValue * standardError;
            double upperBound = model.coefficients().apply(0) + criticalValue * standardError;

            // Виведення результатів
            System.out.println("Confidence Interval for Mean Value:");
            System.out.println("Lower Bound: " + lowerBound);
            System.out.println("Upper Bound: " + upperBound);

            predictions.select("prediction", "PE").show();

            RegressionEvaluator evaluator = new RegressionEvaluator()
                    .setLabelCol("PE")
                    .setPredictionCol("prediction")
                    .setMetricName("mae");

            double mae = evaluator.evaluate(predictions);
            System.out.println("Mean Absolute Error (MAE): " + mae);

            DataVisualizer dataVisualizer = new DataVisualizer(convertedData,model.coefficients().toArray()[0],
                    model.intercept());
            dataVisualizer.makeVisualisation();
        }
    }
}



