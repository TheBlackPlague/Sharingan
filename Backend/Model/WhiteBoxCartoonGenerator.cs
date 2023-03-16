#pragma warning disable CS8981
using TorchSharp;
using TorchSharp.Modules;
using f = TorchSharp.torch.nn.functional;
using static TorchSharp.torch;

namespace Backend.Model;

public class WhiteBoxCartoonGenerator : nn.Module<Tensor, Tensor>
{

    private const string NAME = "Generator";

    private readonly Sequential D0;
    private readonly Sequential D1;
    private readonly Sequential D2;
    private readonly MemorySequential MemoryLine;
    private readonly Sequential U0;
    private readonly Sequential U1;
    private readonly Sequential U2;

    private readonly PaddingModes PaddingType;

    private readonly (string, nn.Module<Tensor, Tensor>)[] Modules;

    public WhiteBoxCartoonGenerator(int channels = 3, int features = 32, int memory = 4,
        PaddingModes paddingType = PaddingModes.Zeros) : base(NAME)
    {
        PaddingType = paddingType;

        Conv2d Conv2DLayer(int inputChannel, int outputChannel, int kernelSize, int stride, int padding)
            => nn.Conv2d(inputChannel, outputChannel, kernelSize, stride, padding, paddingMode: PaddingType);

        LeakyReLU LeakyReLULayer() => nn.LeakyReLU(0.2, true);
        
        nn.Module<Tensor, Tensor> SoftCast(Sequential module) => module;

        D0 = nn.Sequential(
            Conv2DLayer(channels, features, 7, 1, 3),
            LeakyReLULayer()
        );
        D1 = nn.Sequential(
            Conv2DLayer(features, features, 3, 2, 1),
            LeakyReLULayer(),
            Conv2DLayer(features, features * 2, 3, 1, 1),
            LeakyReLULayer()
        );
        D2 = nn.Sequential(
            Conv2DLayer(features * 2, features * 2, 3, 2, 1),
            LeakyReLULayer(),
            Conv2DLayer(features * 2, features * 4, 3, 1, 1),
            LeakyReLULayer()
        );
        MemoryLine = new MemorySequential(
            Enumerable.Range(0, memory)
                .Select(_ => new MemoryLine(features * 4, 3, 1, 1, PaddingType)).ToArray()
        );
        U0 = nn.Sequential(
            Conv2DLayer(features * 4, features * 2, 3, 1, 1),
            LeakyReLULayer()
        );
        U1 = nn.Sequential(
            Conv2DLayer(features * 2, features * 2, 3, 1, 1),
            LeakyReLULayer(),
            Conv2DLayer(features * 2, features, 3, 1, 1),
            LeakyReLULayer()
        );
        U2 = nn.Sequential(
            Conv2DLayer(features, features, 3, 1, 1),
            LeakyReLULayer(),
            Conv2DLayer(features, channels, 7, 1, 3)
        );

        Modules = new[]
        {
            ("d0", SoftCast(D0)),
            ("d1", SoftCast(D1)),
            ("d2", SoftCast(D2)),
            ("memory_line", SoftCast(MemoryLine)),
            ("u0", SoftCast(U0)),
            ("u1", SoftCast(U1)),
            ("u2", SoftCast(U2))
        };
    }
    
    public override Tensor forward(Tensor x)
    {
        Tensor x0 = D0.forward(x);
        Tensor x1 = D1.forward(x0);

        Tensor y = D2.forward(x1);
        y = MemoryLine.forward(y);
        y = U0.forward(y);
        y = f.interpolate(y, scale_factor: new [] { 2d, 2d }, mode: InterpolationMode.Bilinear, align_corners: false);
        y = U1.forward(y + x1);
        y = f.interpolate(y, scale_factor: new [] { 2d, 2d }, mode: InterpolationMode.Bilinear, align_corners: false);
        y = U2.forward(y + x0);

        return tanh(y);
    }

    public override Dictionary<string, Tensor> state_dict(Dictionary<string, Tensor>? destination = null,
        string? prefix = null)
    {
        Dictionary<string, Tensor> stateDictionary = new();

        foreach ((string, nn.Module<Tensor, Tensor>) module in Modules) {
            foreach (KeyValuePair<string, Tensor> pair in module.Item2.state_dict()) {
                stateDictionary.TryAdd(module.Item1 + '.' + pair.Key, pair.Value);
            }
        }
        
        return stateDictionary;
    }

    protected override nn.Module _to(DeviceType deviceType, int deviceIndex = -1)
    {
        base._to(deviceType, deviceIndex);

        // Move everything else to the device and index as well.
        foreach ((string, nn.Module<Tensor, Tensor>) module in Modules) {
            module.Item2.to(deviceType, deviceIndex);
        }
        
        return this;
    }

}