namespace Titanic.Common
{
    using Microsoft.ML.Data;

    public class SurvivalPrediction
    {
        public bool PredictedLabel { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}


