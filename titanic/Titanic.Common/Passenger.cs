namespace Titanic.Common
{
    using Microsoft.ML.Data;

    public class Passenger
    {
        [LoadColumn(0)]
        public int Id { get; set; }

        [LoadColumn(1)]
        public int Survived { get; set; }

        [LoadColumn(2)]
        public int TicketClass { get; set; }

        [LoadColumn(3)]
        public string Name { get; set; }

        [LoadColumn(4)]
        public string Sex { get; set; }

        [LoadColumn(5)]
        public string Age { get; set; }

        [LoadColumn(6)]
        public string SiblingsOrSpouses { get; set; }

        [LoadColumn(7)]
        public string ParentsOrChildren { get; set; }

        [LoadColumn(8)]
        public string TicketNumber { get; set; }

        [LoadColumn(9)]
        public float Fare { get; set; }

        [LoadColumn(10)]
        public string Cabin { get; set; }

        [LoadColumn(11)]
        public string Embarked { get; set; }
    }
}
