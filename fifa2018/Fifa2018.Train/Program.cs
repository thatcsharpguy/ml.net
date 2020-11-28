using System;

namespace Fifa2018.Train
{
    using Microsoft.ML;
    using System.Linq;
    using Fifa2018.Common;
    class Train
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            var data = mlContext.Data.ReadTeamStatistics(args[0]);

            var trainingData = mlContext.Data.TrainTestSplit(data);

            var booleanMap = mlContext.Data.LoadFromEnumerable(new[]
            {
                new { StringValue = "Yes", Value = true },
                new { StringValue = "No", Value = false },
            });
            var transformLabel = mlContext.Transforms.Conversion.MapValue(
                outputColumnName: "Label",
                lookupMap: booleanMap,
                keyColumn: booleanMap.Schema[0],
                valueColumn: booleanMap.Schema[1],
                inputColumnName: nameof(TeamStatistics.ManOfTheMatch)
            );

            var convertGoalScored = mlContext.Transforms.Conversion.ConvertType(
                outputColumnName: nameof(TeamStatistics.GoalScored),
                inputColumnName: nameof(TeamStatistics.GoalScored)
            );

            var convertBallPosession = mlContext.Transforms.Conversion.ConvertType(
                outputColumnName: nameof(TeamStatistics.BallPosession),
                inputColumnName: nameof(TeamStatistics.BallPosession)
            );

            var convertAttempts = mlContext.Transforms.Conversion.ConvertType(
                outputColumnName: nameof(TeamStatistics.Attempts),
                inputColumnName: nameof(TeamStatistics.Attempts)
            );

            var convertOnTarget = mlContext.Transforms.Conversion.ConvertType(
                outputColumnName: nameof(TeamStatistics.OnTarget),
                inputColumnName: nameof(TeamStatistics.OnTarget)
            );


            var scaleFeatures = mlContext.Transforms.NormalizeMeanVariance(
                new InputOutputColumnPair[]
                {
                    new InputOutputColumnPair(nameof(TeamStatistics.GoalScored), nameof(TeamStatistics.GoalScored)),
                    new InputOutputColumnPair(nameof(TeamStatistics.BallPosession), nameof(TeamStatistics.BallPosession)),
                    new InputOutputColumnPair(nameof(TeamStatistics.Attempts), nameof(TeamStatistics.Attempts)),
                    new InputOutputColumnPair(nameof(TeamStatistics.OnTarget), nameof(TeamStatistics.OnTarget)),
                }
            );

            var appendFeatures = mlContext.Transforms.Concatenate(
                outputColumnName: "Features",
                nameof(TeamStatistics.GoalScored),
                nameof(TeamStatistics.BallPosession),
                nameof(TeamStatistics.Attempts),
                nameof(TeamStatistics.OnTarget)
            );


            var classifier = mlContext.BinaryClassification.Trainers.LinearSvm(
                    labelColumnName: "Label"
                );
            var transformPipeline = transformLabel
                .Append(convertOnTarget)
                .Append(convertGoalScored)
                .Append(convertBallPosession)
                .Append(convertAttempts)
                .Append(scaleFeatures)
                .Append(appendFeatures);

            var predictPipeline = transformPipeline.Append(classifier);
            var trainedModel = predictPipeline.Fit(trainingData.TrainSet);

            var metricas = mlContext.BinaryClassification.EvaluateNonCalibrated(
                data: trainedModel.Transform(trainingData.TestSet),
                labelColumnName: "Label"
            );

            Console.WriteLine("Metricas:");
            Console.WriteLine($"\tExactitud: {metricas.Accuracy:0.###}");
            Console.WriteLine($"\tPrecision: {metricas.PositivePrecision:0.###}");
            Console.WriteLine($"\tRecall:    {metricas.PositiveRecall:0.###}");

            mlContext.Model.Save(
                model: trainedModel,
                inputSchema: data.Schema,
                filePath: args[1]
            );
        }
    }
}
