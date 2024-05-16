package kpi.lab2;

import org.knowm.xchart.*;

public class ConfidenceIntervalPlot {

    public static void main(String[] args) {
        // Дані для побудови графіку
        double[] xData = {0.5}; // Значення X
        double[] yData = {15.22}; // Прогнозоване значення Y
        double lowerBound = 10.0; // Нижня межа довірчого інтервалу
        double upperBound = 20.0; // Верхня межа довірчого інтервалу

        // Створення графіку
        XYChart chart = new XYChartBuilder().width(800).height(600).title("Confidence Interval Plot").xAxisTitle("X").yAxisTitle("Y").build();

        // Додавання прогнозованих значень і довірчого інтервалу на графік
        chart.addSeries("Prediction", xData, yData);
        chart.addSeries("Confidence Interval", new double[]{xData[0], xData[0]}, new double[]{lowerBound, upperBound});

        // Відображення графіку
        new SwingWrapper<>(chart).displayChart();
    }
}
