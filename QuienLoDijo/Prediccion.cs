namespace QuienLoDijo
{
    using Microsoft.ML.Data;

    class Prediccion 
    {

        [ColumnName("Personaje")]
        public bool Personaje { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }
}