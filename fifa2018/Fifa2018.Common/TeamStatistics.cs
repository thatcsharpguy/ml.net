namespace Fifa2018.Common
{
    using System;
    using Microsoft.ML.Data;

    public class TeamStatistics
    {
        [LoadColumn(0)]
        public string MatchDate { get; set; }

        [LoadColumn(1)]
        public string Team { get; set; }

        [LoadColumn(2)]
        public string Opponent { get; set; }

        [LoadColumn(3)]
        public int GoalScored { get; set; }

        [LoadColumn(4)]
        public int BallPosession { get; set; }

        [LoadColumn(5)]
        public int Attempts { get; set; }

        [LoadColumn(6)]
        public int OnTarget { get; set; }

        [LoadColumn(7)]
        public int OffTarget { get; set; }

        [LoadColumn(8)]
        public int Blocked { get; set; }

        [LoadColumn(9)]
        public int Corners { get; set; }

        [LoadColumn(10)]
        public int Offsides { get; set; }

        [LoadColumn(11)]
        public int FreeKicks { get; set; }

        [LoadColumn(12)]
        public int Saves { get; set; }

        [LoadColumn(13)]
        public int PassAccuracy { get; set; }

        [LoadColumn(14)]
        public int Passes { get; set; }

        [LoadColumn(15)]
        public int DistanceCovered { get; set; }

        [LoadColumn(16)]
        public int FoulsCommited { get; set; }

        [LoadColumn(17)]
        public int YellowCards { get; set; }

        [LoadColumn(19)]
        public int RedCards { get; set; }

        [LoadColumn(18)]
        public int YellowAndRedCards { get; set; }

        [LoadColumn(20)]
        public string ManOfTheMatch { get; set; }

        [LoadColumn(21)]
        public int FirstGoalTime { get; set; }

        [LoadColumn(22)]
        public string Round { get; set; }

        [LoadColumn(23)]
        public string PenaltyShootout { get; set; }

        [LoadColumn(24)]
        public int PenaltyShootoutGoals { get; set; }

        [LoadColumn(25)]
        public int OwnGoals { get; set; }

        [LoadColumn(26)]
        public int FirstOwnGoalTime { get; set; }

    }
}
