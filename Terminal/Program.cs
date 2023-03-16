// See https://aka.ms/new-console-template for more information

using Backend;
using Backend.Inference;
using Backend.Model;
using OpenCvSharp;
using Spectre.Console;
using Terminal.Spectre;
using TorchSharp;
using Size = OpenCvSharp.Size;

const string modelPath = "Backend.Binary.WhiteBoxCartoonGenerator-8365fa35b8.bin";
const string p1  = "[red]SHARINGAN[/] [yellow]>>[/] 🖥️ Select a device to run the model on.";
const string p2  = "[red]SHARINGAN[/] [yellow]>>[/] 🧠 Loading model...";
const string p3  = "[red]SHARINGAN[/] [yellow]>>[/] 🧠 Model loaded to device: ";
const string p4  = "[red]SHARINGAN[/] [yellow]>>[/] 📁 Input file path: ";
const string p5  = "[red]SHARINGAN[/] [yellow]>>[/] 📁 Output file path: ";
const string p6  = "[red]SHARINGAN[/] [yellow]>>[/] 📂 Generating video file streams...";
const string p7  = "[red]SHARINGAN[/] [yellow]>>[/] 📂 Video file streams generated.";
const string p8  = "[red]SHARINGAN[/] [yellow]>>[/] 🚀 Processing...";
const string p9  = "[yellow]🎞️🔍 Inferring frames[/]";
const string p10 = "[yellow]🎞️💾 Writing frames[/]";
const string p11 = "[red]SHARINGAN[/] [yellow]>>[/] ✅ Processing done!";
const string p12 = "[red]SHARINGAN[/] [yellow]>>[/] 🗑️ Video file streams disposed.";

FourCC codec = FourCC.FromString("mp4v");

(IEnumerable<Mat>, int, double, Size) ReadVideo(string path)
{
    VideoCapture cap = VideoCapture.FromFile(path);
    int length = cap.FrameCount;
    double fps = cap.Fps;
    int h = cap.FrameHeight;
    int w = cap.FrameWidth;

    IEnumerable<Mat> Frames()
    {
        while (true) {
            Mat frame = new();
            bool ret = cap.Read(frame);
            if (ret) yield return frame;
            else {
                cap.Release();
                break;
            }
        }
    }

    return (Frames(), length, fps, new Size(w, h));
}

int DetermineBatchSize(Size frameDimension)
{
    return 3840 * 2160 / (frameDimension.Width * frameDimension.Height);
}

torch.Device device = args.Length > 0 ? args[0] : AnsiConsole.Prompt(
    new SelectionPrompt<torch.Device>()
        .PageSize(4)
        .Title(p1)
        .AddChoices(new torch.Device("cpu"), new torch.Device("cuda"))
);

AnsiConsole.MarkupLine(p2);
WhiteBoxCartoonGenerator model = new();
Utility.LoadModel(model, modelPath);
model.to(device);
AnsiConsole.MarkupLine(p3 + device);

string input = args.Length > 1 ? 
    args[1] : AnsiConsole.Ask<string>(p4).Replace("\"", "");
string output = args.Length > 2 ? 
    args[2] : AnsiConsole.Ask<string>(p5).Replace("\"", "");

AnsiConsole.MarkupLine(p6);
(IEnumerable<Mat> frames, int length, double fps, Size size) = ReadVideo(input);
VideoWriter? writer = null;
AnsiConsole.MarkupLine(p7);

int batchSize = DetermineBatchSize(size);

AnsiConsole.MarkupLine(p8);

AnsiConsole.Progress()
    .Columns(
        new TaskDescriptionColumn
        {
            Alignment = Justify.Left
        },
        new ProgressBarColumn
        {
            Width = 40,
            CompletedStyle = new Style(Color.LightGreen, decoration: Decoration.Bold),
            RemainingStyle = new Style(Color.White, decoration: Decoration.Dim),
            FinishedStyle = new Style(Color.Green)
        },
        new PercentageColumn(),
        new SpinnerColumn(new MoonSpinner()),
        new ElapsedTimeColumn(),
        new FpsColumn
        {
            BatchSize = batchSize 
        }
    ).AutoClear(true).Start(ProgressContext =>
    {
        ProgressTask inferTask = ProgressContext.AddTask(
            p9, 
            true, 
            length / batchSize
        );
        ProgressTask writeTask = ProgressContext.AddTask(
            p10, 
            true, 
            length / batchSize
        );
    
        using (torch.no_grad()) foreach (IEnumerable<Mat> frameChunk in frames.Chunk(batchSize)) {
            IEnumerable<Mat> inferredFrames = model.InferFrames(frameChunk, size, device);
            GC.Collect();
            inferTask.Increment(1);
        
            foreach (Mat frame in inferredFrames) {
                writer ??= new VideoWriter(output, codec, fps, frame.Size());
                writer.Write(frame);
            }
            GC.Collect();
            writeTask.Increment(1);
        } 
    });

AnsiConsole.MarkupLine(p11);

writer?.Release();

AnsiConsole.MarkupLine(p12);