namespace WhatCardIsThis.Objects
{
    using Microsoft.ML.Data;
    public class CardPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Type { get; set; }

    }
}