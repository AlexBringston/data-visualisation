package kpi.lab2;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.awt.Color;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataVisualizer {

    private final Dataset<Row> data;
    private final double coefficient;
    private final double intercept;

    public DataVisualizer(Dataset<Row> data, double coefficient, double intercept) {
        this.data = data;
        this.coefficient = coefficient;
        this.intercept = intercept;
    }

    public void makeVisualisation() {
        List<Double> temperatures = data.select("AT").as(Encoders.DOUBLE()).collectAsList();
        List<Double> powers = data.select("PE").as(Encoders.DOUBLE()).collectAsList();
        List<Double> linearRegressionGraph = new ArrayList<>();

        temperatures.stream()
                .map(temperature -> coefficient * temperature + intercept)
                .forEach(linearRegressionGraph::add);

        XYChart diagram = new XYChart(1200, 600);
        diagram.setTitle("Залежність вироблення енергії від температури навколишнього середовища");
        diagram.setXAxisTitle("Температура");
        diagram.setYAxisTitle("Годинне вироблення енергії");

        diagram.addSeries("Вироблена енергія\n",
                        temperatures, powers)
                .setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter)
                .setMarkerColor(Color.BLUE)
                .setLineStyle(SeriesLines.NONE);

        XYSeries newSeries = diagram.addSeries(String.format("Графік лінійної регресії \nY = %.2fx + %.2f",
                coefficient, intercept), temperatures, linearRegressionGraph);
        newSeries.setMarker(SeriesMarkers.NONE);
        newSeries.setLineColor(Color.ORANGE);

        try {
            BitmapEncoder.saveBitmap(diagram, "result.png", BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
