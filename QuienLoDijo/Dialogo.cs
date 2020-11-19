namespace QuienLoDijo
{
    using Microsoft.ML.Data;

    class Dialogo 
    {
        [LoadColumn(4)]
        public string Texto { get; set; }

        [LoadColumn(1)]
        public bool Personaje { get; set; }
    }
}