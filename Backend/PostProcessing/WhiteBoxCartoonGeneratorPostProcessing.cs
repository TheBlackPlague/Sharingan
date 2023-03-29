#pragma warning disable CS8981
using f = TorchSharp.torch.nn.functional;
using static TorchSharp.torch;

namespace Backend.PostProcessing;

public static class WhiteBoxCartoonGeneratorPostProcessing
{

    public static Tensor PostProcess(Tensor x, Tensor y, int r = 1, double ep = 1e-2)
    {
        (long h, long w) = (x.shape[2], x.shape[3]);

        Tensor n = SquareRegionFilter(ones(1, 1, h, w, x.dtype, x.device), r);

        Tensor mX = SquareRegionFilter(x, r) / n;
        Tensor mY = SquareRegionFilter(y, r) / n;
        Tensor cXY = SquareRegionFilter(x * y, r) / n - mX * mY;
        Tensor cXX = SquareRegionFilter(x * x, r) / n - mX * mX;

        Tensor a = cXY / (cXX + ep);
        Tensor b = mY - a * mX;

        Tensor mA = SquareRegionFilter(a, r) / n;
        Tensor mB = SquareRegionFilter(b, r) / n;

        return mA * x + mB;
    }

    private static Tensor SquareRegionFilter(Tensor x, int r)
    {
        int channel = (int)x.shape[1];
        int kernel = 2 * r + 1;
        float weight = 1f / MathF.Pow(kernel, 2);
        Tensor regionWeight = 
            weight * ones(channel, 1, kernel, kernel, ScalarType.Float32, x.device);
        return f.conv2d(x, regionWeight, strides: new[] { 1L, 1L }, padding: new[] { (long)r, r }, groups: channel);
    }

}