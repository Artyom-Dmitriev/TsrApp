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
        // Tolerate reading an old log that lacks the milestone-3 columns; the
        // missing fields just stay at their defaults instead of throwing.
        HeaderValidated = null,
        MissingFieldFound = null,
    };

    private readonly string _path;
    private readonly object _lock = new();

    public PredictionLogger(string path)
    {
        _path = path;
        string? dir = Path.GetDirectoryName(_path);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        MigrateIfNeeded();
    }

    /// <summary>
    /// One-time upgrade of a pre-existing 5-column log to the new 9-column
    /// schema. Old rows are preserved with the new fields left empty. Without
    /// this, appending 9-field rows under a 5-column header would corrupt the
    /// CSV. Idempotent: a log that already has the new header is left untouched.
    /// </summary>
    private void MigrateIfNeeded()
    {
        lock (_lock)
        {
            if (!File.Exists(_path)) return;

            string? header;
            using (var reader = new StreamReader(_path))
                header = reader.ReadLine();

            if (header is null) return;                  // empty file
            if (header.Contains("Mode")) return;         // already new schema

            List<PredictionLogEntry> records;
            using (var reader = new StreamReader(_path))
            using (var csv = new CsvReader(reader, Cfg))
                records = csv.GetRecords<PredictionLogEntry>().ToList();

            using var writer = new StreamWriter(_path, append: false);
            using var outCsv = new CsvWriter(writer, Cfg);
            outCsv.WriteHeader<PredictionLogEntry>();
            outCsv.NextRecord();
            foreach (var e in records)
            {
                outCsv.WriteRecord(e);
                outCsv.NextRecord();
            }
        }
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
