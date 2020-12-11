

namespace Titanic.Production
{
    using System;
    using System.Linq;
    using System.Collections.Generic;
    using Microsoft.ML;
    using Titanic.Common;

    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var trainedModel = context.Model.Load("model.zip", out var schema);

            var predictionEngine = context.Model.CreatePredictionEngine<Passenger, SurvivalPrediction>(
                transformer: trainedModel
            );

            var newPassenger = new Passenger
            {
                TicketClass = 1,
                Sex = "male",
                Embarked = "S",
                Fare = 11.5f
            };

            var prediction = predictionEngine.Predict(newPassenger);

            var survivalStatus = prediction.PredictedLabel ? "sobrevivió" : "no sobrevivió";

            Console.WriteLine($"El modelo dice que el pasajero... " + survivalStatus + ".");
        }
    }
}
