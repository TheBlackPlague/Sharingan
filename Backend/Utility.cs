using System.Reflection;
using System.Text;
using OpenCvSharp;
using Size = OpenCvSharp.Size;
using static TorchSharp.torch;

namespace Backend;

public static class Utility
{

    private static readonly Size Base = new(256, 256);

    public static string Shape(this Tensor x)
    {
        StringBuilder sb = new();
        sb.Append('(');
        bool first = true;
        foreach (long n in x.shape) {
            if (first) first = false;
            else sb.Append(',').Append(' ');

            sb.Append(n);
        }
        sb.Append(')');
        return sb.ToString();
    }

    public static string Detail(this Tensor x)
    {
        StringBuilder sb = new();
        sb.Append(x.Shape());
        sb.Append(" [Device = ").Append(x.device).Append(", DataType = ").Append(x.dtype).Append(']');
        return sb.ToString();
    }
    
    public static string Shape(this Mat x)
    {
        StringBuilder sb = new();
        sb.Append('(').Append(x.Height).Append(',').Append(' ').Append(x.Width).Append(',').Append(' ')
            .Append(x.Channels()).Append(')');
        return sb.ToString();
    }

    public static Tensor ImageToTensor(Mat image, Size size)
    {
        size = ScaleBase(size, 32);
        image = image.Resize(size);
        image.ToArray(out byte[,,] data);
        return from_array(data, ScalarType.Byte, "cuda")
            .permute(2, 0, 1)
            .NormalizeImageTensor(0.5, 0.5, 255);
    }

    public static void LoadModel<T>(T model, string pathToModel) where T : nn.Module<Tensor, Tensor>
    {
        using Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(pathToModel)!;
        model.load(stream);
    }

    private static Size ScalarReduction(Size size, Size minSize, int scale)
    {
        int height;
        int width;
        
        if (size.Height <= minSize.Height) height = minSize.Height;
        else height = size.Height - size.Height % scale;

        if (size.Width <= minSize.Width) width = minSize.Width;
        else width = size.Width - size.Width % scale;

        return new Size(width, height);
    }

    private static Size ScaleBase(Size size, int scale)
    {
        return ScalarReduction(size, Base, scale);
    }

    private static Tensor NormalizeImageTensor(this Tensor x, double meanValue, double stdValue, int maxPixelValue)
    {
        return (x.to_type(float32) - meanValue * maxPixelValue) / (stdValue * maxPixelValue);
    }

    private static void ToArray(this Mat image, out byte[,,] data)
    {
        byte[,,] dataTemp = new byte[image.Height, image.Width, 3];

        Parallel.For(0, image.Width, w =>
        {
            for (int h = 0; h < image.Height; h++) {
                Vec3b vec = image.At<Vec3b>(h, w);
                dataTemp[h, w, 0] = vec.Item0;
                dataTemp[h, w, 1] = vec.Item1;
                dataTemp[h, w, 2] = vec.Item2;
            }
        });

        data = dataTemp;
    }

}