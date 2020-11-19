namespace WhatCardIsThis.Objects
{
    using Microsoft.ML.Data;
    public class Card
    {
        [LoadColumn(0)]
        public string Id { get; set; }

        [LoadColumn(2)]
        public string Type { get; set; }

        [LoadColumn(3)]
        public string Description { get; set; }
    }
}