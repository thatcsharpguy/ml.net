

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


var data = new [] {
	new { TicketClass = 1, Embarked = "Q", Sex = "female", Fare = 4 },
	new { TicketClass = 1, Embarked = "Q", Sex = "female", Fare = 10 },
	new { TicketClass = 1, Embarked = "S", Sex = "male", Fare = 11 },
	new { TicketClass = 0, Embarked = "Q", Sex = "female", Fare = 2 },
	new { TicketClass = 3, Embarked = "C", Sex = "female", Fare = 8 },
	new { TicketClass = 3, Embarked = "C", Sex = "male", Fare = 11 },
	new { TicketClass = 1, Embarked = "Q", Sex = "female", Fare = 6 },
	new { TicketClass = 1, Embarked = "C", Sex = "male", Fare = 6 },
	new { TicketClass = 3, Embarked = "C", Sex = "male", Fare = 15 },
	new { TicketClass = 4, Embarked = "C", Sex = "male", Fare = 15 },
	new { TicketClass = 0, Embarked = "S", Sex = "male", Fare = 1 },
	new { TicketClass = 1, Embarked = "Q", Sex = "male", Fare = 11 },
	new { TicketClass = 1, Embarked = "S", Sex = "male", Fare = 1 },
	new { TicketClass = 2, Embarked = "Q", Sex = "female", Fare = 7 },
	new { TicketClass = 0, Embarked = "S", Sex = "female", Fare = 10 },
	new { TicketClass = 2, Embarked = "Q", Sex = "female", Fare = 14 },
	new { TicketClass = 1, Embarked = "Q", Sex = "female", Fare = 6 },
	new { TicketClass = 2, Embarked = "S", Sex = "male", Fare = 7 },
	new { TicketClass = 0, Embarked = "S", Sex = "male", Fare = 8 },
	new { TicketClass = 2, Embarked = "C", Sex = "male", Fare = 2 },
	new { TicketClass = 0, Embarked = "Q", Sex = "male", Fare = 14 },
	new { TicketClass = 2, Embarked = "Q", Sex = "male", Fare = 13 },
	new { TicketClass = 4, Embarked = "S", Sex = "male", Fare = 15 },
	new { TicketClass = 0, Embarked = "S", Sex = "female", Fare = 2 },
	new { TicketClass = 4, Embarked = "S", Sex = "male", Fare = 11 },
	new { TicketClass = 2, Embarked = "C", Sex = "male", Fare = 4 },
	new { TicketClass = 4, Embarked = "S", Sex = "male", Fare = 5 },
	new { TicketClass = 0, Embarked = "C", Sex = "female", Fare = 9 },
	new { TicketClass = 1, Embarked = "Q", Sex = "male", Fare = 4 },
	new { TicketClass = 4, Embarked = "C", Sex = "male", Fare = 2 },
	new { TicketClass = 4, Embarked = "Q", Sex = "female", Fare = 8 },
	new { TicketClass = 4, Embarked = "S", Sex = "male", Fare = 2 },
	new { TicketClass = 3, Embarked = "Q", Sex = "male", Fare = 10 },
	new { TicketClass = 3, Embarked = "C", Sex = "female", Fare = 4 },
	new { TicketClass = 3, Embarked = "Q", Sex = "male", Fare = 3 },
	new { TicketClass = 0, Embarked = "Q", Sex = "female", Fare = 10 },
	new { TicketClass = 4, Embarked = "S", Sex = "female", Fare = 6 },
	new { TicketClass = 4, Embarked = "S", Sex = "female", Fare = 12 },
	new { TicketClass = 3, Embarked = "S", Sex = "female", Fare = 1 },
	new { TicketClass = 1, Embarked = "S", Sex = "male", Fare = 7 },
	new { TicketClass = 0, Embarked = "Q", Sex = "male", Fare = 8 },
	new { TicketClass = 4, Embarked = "Q", Sex = "male", Fare = 1 },
	new { TicketClass = 4, Embarked = "S", Sex = "male", Fare = 5 },
	new { TicketClass = 0, Embarked = "Q", Sex = "female", Fare = 12 },
	new { TicketClass = 0, Embarked = "Q", Sex = "female", Fare = 13 },
	new { TicketClass = 4, Embarked = "S", Sex = "male", Fare = 15 },
	new { TicketClass = 0, Embarked = "C", Sex = "male", Fare = 11 },
	new { TicketClass = 1, Embarked = "C", Sex = "male", Fare = 3 },
	new { TicketClass = 3, Embarked = "S", Sex = "female", Fare = 6 },
	new { TicketClass = 2, Embarked = "C", Sex = "male", Fare = 11 },
	new { TicketClass = 0, Embarked = "Q", Sex = "male", Fare = 12 },
	new { TicketClass = 3, Embarked = "C", Sex = "female", Fare = 4 },
	new { TicketClass = 0, Embarked = "C", Sex = "female", Fare = 10 },
	new { TicketClass = 0, Embarked = "C", Sex = "female", Fare = 2 },
	new { TicketClass = 3, Embarked = "C", Sex = "male", Fare = 7 },
	new { TicketClass = 1, Embarked = "S", Sex = "male", Fare = 12 },
	new { TicketClass = 4, Embarked = "S", Sex = "female", Fare = 4 },
	new { TicketClass = 0, Embarked = "S", Sex = "female", Fare = 9 },
	new { TicketClass = 0, Embarked = "Q", Sex = "male", Fare = 5 },
	new { TicketClass = 2, Embarked = "C", Sex = "female", Fare = 15 },
	new { TicketClass = 1, Embarked = "S", Sex = "male", Fare = 10 },
	new { TicketClass = 3, Embarked = "S", Sex = "male", Fare = 4 },
	new { TicketClass = 1, Embarked = "Q", Sex = "female", Fare = 4 },
	new { TicketClass = 3, Embarked = "S", Sex = "female", Fare = 4 },
	new { TicketClass = 3, Embarked = "C", Sex = "female", Fare = 14 },
	new { TicketClass = 0, Embarked = "S", Sex = "female", Fare = 5 },
	new { TicketClass = 0, Embarked = "C", Sex = "male", Fare = 10 },
	new { TicketClass = 0, Embarked = "C", Sex = "male", Fare = 13 },
	new { TicketClass = 1, Embarked = "Q", Sex = "female", Fare = 1 },
	new { TicketClass = 4, Embarked = "C", Sex = "female", Fare = 6 },
	new { TicketClass = 0, Embarked = "S", Sex = "male", Fare = 7 },
	new { TicketClass = 0, Embarked = "C", Sex = "female", Fare = 8 },
	new { TicketClass = 3, Embarked = "S", Sex = "female", Fare = 2 },
	new { TicketClass = 3, Embarked = "S", Sex = "female", Fare = 8 },
	new { TicketClass = 2, Embarked = "S", Sex = "female", Fare = 10 },
	new { TicketClass = 1, Embarked = "C", Sex = "female", Fare = 1 },
	new { TicketClass = 0, Embarked = "S", Sex = "female", Fare = 2 },
	new { TicketClass = 3, Embarked = "Q", Sex = "male", Fare = 2 },
	new { TicketClass = 2, Embarked = "S", Sex = "male", Fare = 11 },
	new { TicketClass = 0, Embarked = "Q", Sex = "male", Fare = 6 },
	new { TicketClass = 0, Embarked = "Q", Sex = "female", Fare = 2 },
	new { TicketClass = 3, Embarked = "Q", Sex = "female", Fare = 8 },
	new { TicketClass = 4, Embarked = "Q", Sex = "male", Fare = 15 },
	new { TicketClass = 1, Embarked = "C", Sex = "female", Fare = 10 },
	new { TicketClass = 0, Embarked = "Q", Sex = "male", Fare = 9 },
	new { TicketClass = 1, Embarked = "Q", Sex = "male", Fare = 7 },
	new { TicketClass = 3, Embarked = "C", Sex = "male", Fare = 12 },
	new { TicketClass = 2, Embarked = "Q", Sex = "female", Fare = 3 },
	new { TicketClass = 2, Embarked = "S", Sex = "female", Fare = 15 },
	new { TicketClass = 3, Embarked = "C", Sex = "female", Fare = 2 },
	new { TicketClass = 1, Embarked = "S", Sex = "male", Fare = 2 },
	new { TicketClass = 4, Embarked = "S", Sex = "female", Fare = 4 },
	new { TicketClass = 1, Embarked = "C", Sex = "female", Fare = 1 },
	new { TicketClass = 4, Embarked = "S", Sex = "female", Fare = 7 },
	new { TicketClass = 1, Embarked = "C", Sex = "female", Fare = 5 },
	new { TicketClass = 1, Embarked = "C", Sex = "male", Fare = 7 },
	new { TicketClass = 0, Embarked = "C", Sex = "male", Fare = 5 },
	new { TicketClass = 2, Embarked = "C", Sex = "female", Fare = 8 },
	new { TicketClass = 1, Embarked = "Q", Sex = "male", Fare = 12 },
	new { TicketClass = 4, Embarked = "Q", Sex = "female", Fare = 4 }
};

            var predictions = new List<SurvivalPrediction>();

            foreach (var o in data)
            {
                SurvivalPrediction se = null;
                predictionEngine.Predict(new Passenger
                {
                    TicketClass = o.TicketClass,
                    Embarked = o.Embarked,
                    Sex = o.Sex,
                    Fare = o.Fare
                }, ref se);

                predictions.Add(se);
            }

            foreach(var pred in predictions.OrderBy(pred => pred.Probability))
            {
                Console.WriteLine(pred.Label);
                Console.WriteLine(pred.Probability);
            }
        }
    }
}
