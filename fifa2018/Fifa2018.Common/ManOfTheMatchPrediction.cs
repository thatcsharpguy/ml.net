namespace Fifa2018.Common
{
    using Microsoft.ML.Data;
    public class ManOfTheMatchPrediction
    {
        [ColumnName("Label")]
        public bool ManOfTheMatch { get; set; }

        [ColumnName(nameof(ManOfTheMatchPrediction.Probability))]
        public float Probability { get; set; }

        [ColumnName(nameof(ManOfTheMatchPrediction.Score))]
        public float Score { get; set; }
    }
}
