namespace OpenCVDemo.Services;

public class EastOpenCvServiceConfiguration
{
    public string ModelPath { get; set; }
    public int NumThreads { get; set; }
    public float ConfidenceThreshold { get; set; }

    public float NMSThreshold { get; set; }
    public float FrameSimilarityThreshold { get; set; }
    public bool UseGpuAcceleration { get; set; }
}