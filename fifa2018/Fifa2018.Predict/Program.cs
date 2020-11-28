using System;

namespace Fifa2018.Predict
{
    using Fifa2018.Common;
    using Microsoft.ML;
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var modelo = mlContext.Model.Load(args[0], out var schema);
            var predictor = mlContext.Model.CreatePredictionEngine<TeamStatistics, ManOfTheMatchPrediction>(modelo);


            for ( int i=0; i < 10; i++ )
            {
                var statistics = new TeamStatistics
                {
                    GoalScored = i
                };

                var prediction = predictor.Predict(statistics);

                Console.WriteLine($"Prediction for goals {statistics.GoalScored} {prediction.ManOfTheMatch} - {prediction.Score} - {prediction.Probability}");
            }
        }
    }
}
