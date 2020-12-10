namespace Titanic.Train
{
    using System;
    using Microsoft.ML;
    using Titanic.Common;
    using static Microsoft.ML.Transforms.MissingValueReplacingEstimator;

    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext(seed: 42);
            IDataView allData = context.Data.LoadFromTextFile<Passenger>(
                path: "train.csv",
                separatorChar: ',',
                hasHeader: true
            );

            var splits = context.Data.TrainTestSplit(
                data: allData,
                testFraction: 0.2
            );

            var booleanMap = context.Data.LoadFromEnumerable(new[]
            {
                new { InputValue = 1, Value = true },
                new { InputValue = 2, Value = false },
            });

            var transformLabel = context.Transforms.Conversion.MapValue(
                outputColumnName: "Label",
                lookupMap: booleanMap,
                keyColumn: booleanMap.Schema["InputValue"],
                valueColumn: booleanMap.Schema["Value"],
                inputColumnName: nameof(Passenger.Survived)
            );


            var transformTicketClassOneHot = context.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "OneHotTicketClass",
                inputColumnName: nameof(Passenger.TicketClass)
            );

            var transformSeveralOneHot = context.Transforms.Categorical.OneHotEncoding(
                new InputOutputColumnPair []
                {
                    new InputOutputColumnPair(inputColumnName: "Embarked", outputColumnName: "OneHotEmbarked"),
                    new InputOutputColumnPair(inputColumnName: "Sex", outputColumnName: "OneHotSex"),
                }
            );

            var transformFillMeanFare = context.Transforms.ReplaceMissingValues(
                outputColumnName: "Fare",
                inputColumnName: "Fare",
                replacementMode: ReplacementMode.Mean
            );

            var transformNormaliseFare = context.Transforms.NormalizeMinMax(
                outputColumnName: "NormalisedFare",
                inputColumnName: "Fare"
            );

            var transformConcatenateFeatures = context.Transforms.Concatenate(
                outputColumnName: "Features",
                "NormalisedFare", "OneHotTicketClass", "OneHotEmbarked", "OneHotSex"
            );

            var trainer = context.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName:"Label",
                featureColumnName:"Features"
            );

            var pipeline = transformLabel
                .Append(transformTicketClassOneHot)
                .Append(transformSeveralOneHot)
                .Append(transformFillMeanFare)
                .Append(transformNormaliseFare)
                .Append(transformConcatenateFeatures)
                .Append(trainer);

            var trainedModel = pipeline.Fit(splits.TrainSet);

            var metrics = context.BinaryClassification.EvaluateNonCalibrated(
                data: trainedModel.Transform(splits.TestSet),
                labelColumnName: "Label"
            );

            Console.WriteLine($"Exactitud: {metrics.Accuracy:0.##}");
            Console.WriteLine($"Precisión: {metrics.PositivePrecision:0.##}");
            Console.WriteLine($"Recall:    {metrics.PositiveRecall:0.##}");

            context.Model.Save(
                model: trainedModel,
                inputSchema: allData.Schema,
                filePath: "model.zip"
            );
        }
    }
}
