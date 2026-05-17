using System.Globalization;
using System.IO;
using CsvHelper;
using CsvHelper.Configuration;

namespace TsrApp.Services;

public sealed class PredictionLogger
{
    private static readonly CsvConfiguration Cfg = new(CultureInfo.InvariantCulture)
    {
        HasHeaderRecord = true,
    };

    private readonly string _path;
    private readonly object _lock = new();

    public PredictionLogger(string path)
    {
        _path = path;
        string? dir = Path.GetDirectoryName(_path);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);
    }

    public void Append(PredictionLogEntry entry)
    {
        lock (_lock)
        {
            bool isNew = !File.Exists(_path);
            using var writer = new StreamWriter(_path, append: true);
            using var csv = new CsvWriter(writer, Cfg);
            if (isNew)
            {
                csv.WriteHeader<PredictionLogEntry>();
                csv.NextRecord();
            }
            csv.WriteRecord(entry);
            csv.NextRecord();
        }
    }

    public void Clear()
    {
        lock (_lock)
        {
            using var writer = new StreamWriter(_path, append: false);
            using var csv = new CsvWriter(writer, Cfg);
            csv.WriteHeader<PredictionLogEntry>();
            csv.NextRecord();
        }
    }

    public IReadOnlyList<PredictionLogEntry> ReadAll()
    {
        lock (_lock)
        {
            if (!File.Exists(_path)) return Array.Empty<PredictionLogEntry>();
            using var reader = new StreamReader(_path);
            using var csv = new CsvReader(reader, Cfg);
            return csv.GetRecords<PredictionLogEntry>().ToList();
        }
    }
}
