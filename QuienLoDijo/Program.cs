

namespace QuienLoDijo
{
    using System;
    using Microsoft.ML;
    using Microsoft.ML.Transforms.Text;

    class Program
    {
        static void Main(string[] args)
        {
            if(args.Length > 0)
            {
                var mlContext = new MLContext();

                PredictionEngine<Dialogo, Prediccion> predictionEngine;
                ITransformer trainedModel;
                IDataView dataView;

                if(args[0] == "train")
                {
                    // dotnet run train train.csv model.zip

                    dataView = mlContext.Data.LoadFromTextFile<Dialogo>(args[1], hasHeader: true, separatorChar: ',');

                    var dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

                    // var mapLabelsToValues = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName:"Label", inputColumnName: nameof(Dialogo.Personaje));

                    var featurizeOptions = new TextFeaturizingEstimator.Options()
                    {
                        StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() 
                        {
                            Language = TextFeaturizingEstimator.Language.Spanish 
                        },
                        CaseMode = TextNormalizingEstimator.CaseMode.Lower,
                        WordFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 3, UseAllLengths = true },
                        CharFeatureExtractor = new WordBagEstimator.Options() { NgramLength = 4, UseAllLengths= false },
                    };

                    var featuriseText = mlContext.Transforms.Text.FeaturizeText(
                        options: featurizeOptions,
                        outputColumnName: "Features", 
                        inputColumnNames: nameof(Dialogo.Texto));

                    var estimator = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(Prediccion.Personaje), featureColumnName: "Features");

                    // var mapValueToLabels = mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: nameof(Prediccion.Personaje));

                    var pipeline = featuriseText
                        .Append(estimator);

                    trainedModel = pipeline.Fit(dataSplit.TrainSet);

                    predictionEngine = mlContext.Model.CreatePredictionEngine<Dialogo, Prediccion>(trainedModel);
                    
                    mlContext.Model.Save(trainedModel, dataView.Schema, args[2]);

                    var testMetrics = mlContext.BinaryClassification.Evaluate(trainedModel.Transform((dataSplit.TestSet)), labelColumnName: nameof(Prediccion.Personaje));


                    Console.WriteLine($"*************************************************************************************************************");
                    Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
                    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
                    Console.WriteLine($"*       Accuracy:         {testMetrics.Accuracy:0.###}");
                    Console.WriteLine($"*       F1 Score:         {testMetrics.F1Score:0.###}");
                    Console.WriteLine($"*       PosPrecision:     {testMetrics.PositivePrecision:0.###}");
                    Console.WriteLine($"*       NegPrecision:     {testMetrics.NegativePrecision:0.###}");
                    Console.WriteLine($"*       NegRecall:        {testMetrics.NegativeRecall:0.###}");
                    Console.WriteLine($"*       PosRecall:        {testMetrics.PositiveRecall:0.###}");
                    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
                    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
                    Console.WriteLine($"*************************************************************************************************************");

                }
                else if(args[0] == "test")
                {
                    // dotnet run test model.zip "La mafia del poder"

                    trainedModel = mlContext.Model.Load(args[1], out var inputSchema);
                    predictionEngine = mlContext.Model.CreatePredictionEngine<Dialogo, Prediccion>(trainedModel);


                    var prediccion = predictionEngine.Predict(new Dialogo {
                        Texto = args[2]
                    });
                    var personaje = prediccion.Probability > 0.5 ? "López Obrador" : "López-Gatell";
                    Console.WriteLine($"\"{args[2]}\" suena a que lo dijo {personaje}. Score {prediccion.Score} - {prediccion.Probability}");
                }
            }
            else
            {
                Console.WriteLine("Elige una operación: train o test");
            }
        }
    }
}
