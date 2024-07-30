using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using OpenCVDemo.Services;
using OpenCVDemo.ViewModels;

namespace OpenCVDemo
{
    public static class MauiProgram
    {
        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                });


            // Load configuration
            var configBuilder = new ConfigurationBuilder()
                .SetBasePath(AppContext.BaseDirectory)
                .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);

            var config = configBuilder.Build();


            // Register your services here
            builder.Services.Configure<EastOpenCvServiceConfiguration>(config.GetSection("EastOpenCvServiceConfiguration"));
            builder.Services.Configure<TextBoxPlusPlusOpenCvServiceConfiguration>(config.GetSection("TextBoxPlusPlusOpenCvServiceConfiguration"));
            builder.Services.Configure<TextDetectorOpenCvServiceConfiguration>(config.GetSection("TextDetectorOpenCvServiceConfiguration"));
            builder.Services.AddSingleton<IVideoProcessingService, EastOpenCvService>();
            builder.Services.AddSingleton<EastOpenCVProcessingViewModel>();
            builder.Services.AddTransient<VideoProcessingPage>();

#if DEBUG
            builder.Logging.AddDebug();
#endif

            return builder.Build();
        }
    }
}
