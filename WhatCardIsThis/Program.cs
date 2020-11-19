namespace WhatCardIsThis
{
    using Microsoft.ML;
    using System;
    using WhatCardIsThis.Objects;

    class Program
    {
        const string DataInputPath = @"C:\Users\anton\github\ml.net\WhatCardIsThis\Data\cards.csv";
        const string ModelPath = @"C:\Users\anton\github\ml.net\WhatCardIsThis\model.zip";


        private static MLContext _mlContext;
        private static PredictionEngine<Card, CardPrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(Card.Type))
                .Append(_mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(Card.Description)))
                .AppendCacheCheckpoint(_mlContext);

            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(trainingDataView);
            _predEngine = _mlContext.Model.CreatePredictionEngine<Card, CardPrediction>(_trainedModel);

            return trainingPipeline;
        }

        static void Main(string[] args)
        {
            _mlContext = new MLContext();

            // Read data
            _trainingDataView = _mlContext.Data.LoadFromTextFile<Card>(DataInputPath, hasHeader: true, separatorChar: ',');
            var dataSplit = _mlContext.Data.TrainTestSplit(_trainingDataView, testFraction: 0.2);

            var pipeline = ProcessData();

            var trainingPipeline = BuildAndTrainModel(dataSplit.TrainSet, pipeline);

            var aa =_predEngine.Predict(new Card { Description = "This legendary dragon is a powerful engine of destruction. Virtually invincible, very few have faced this awesome creature and lived to tell the tale." });

            _mlContext.Model.Save(_trainedModel, _trainingDataView.Schema, ModelPath);

            Console.WriteLine();
        }
    }
}
