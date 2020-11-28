using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Fifa2018.Common
{
    using Microsoft.ML;
    public static class TeamStatsReaderExtensions
    {
        public static IDataView ReadTeamStatistics(this DataOperationsCatalog data, string file)
        {
            return data.LoadFromTextFile<TeamStatistics>(file, separatorChar: ',', hasHeader: true);
        }
    }
}
