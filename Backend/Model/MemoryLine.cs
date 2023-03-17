using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Backend.Model;

public class MemoryLine : nn.Module<Tensor, Tensor>
{

    private const string NAME = "MemoryLine";

    private readonly Sequential Memory;
    
    private readonly (string, nn.Module<Tensor, Tensor>)[] Modules;

    public MemoryLine(int channels, int kernel, int stride, int padding, PaddingModes paddingType) : base(NAME)
    {
        nn.Module<Tensor, Tensor> SoftCast(Sequential module) => module;
        
        Memory = nn.Sequential(
            nn.Conv2d(channels, channels, kernel, stride, padding, paddingMode: paddingType),
            nn.LeakyReLU(0.2, true),
            nn.Conv2d(channels, channels, kernel, stride, padding, paddingMode: paddingType)
        );
        
        Modules = new[]
        {
            ("memory", SoftCast(Memory))
        };
    }

    public override Tensor forward(Tensor x) => x + Memory.forward(x);

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
        foreach ((_, nn.Module<Tensor, Tensor> module) in Modules) {
            module.to(deviceType, deviceIndex);
        }
        
        return this;
    }

    public override void train(bool train = true)
    {
        base.train(train);

        foreach ((_, nn.Module<Tensor, Tensor> module) in Modules) {
            module.train(train);
        }
    }

}