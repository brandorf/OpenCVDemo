namespace OpenCVDemo.Services;

public class TextDetectorOpenCvServiceConfiguration
{
    public string ModelPath { get; set; }
    public string PrototextPath { get; set; }
    public int NumThreads { get; set; }
    public float ConfidenceThreshold { get; set; }

    public float NMSThreshold { get; set; }
}