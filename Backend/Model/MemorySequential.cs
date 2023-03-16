using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Backend.Model;

public class MemorySequential : Sequential
{

    private readonly MemoryLine[] Modules;

    public MemorySequential(params MemoryLine[] modules) : base(modules)
    {
        Modules = modules;
    }
    
    public override Dictionary<string, Tensor> state_dict(Dictionary<string, Tensor>? destination = null,
        string? prefix = null)
    {
        Dictionary<string, Tensor> stateDictionary = new();

        for (int i = 0; i < Modules.Length; i++) {
            foreach (KeyValuePair<string, Tensor> pair in Modules[i].state_dict()) {
                stateDictionary.TryAdd(i + ('.' + pair.Key), pair.Value);
            }
        }

        return stateDictionary;
    }
    
    protected override nn.Module _to(DeviceType deviceType, int deviceIndex = -1)
    {
        base._to(deviceType, deviceIndex);

        // Move everything else to the device and index as well.
        foreach (MemoryLine line in Modules) {
            line.to(deviceType, deviceIndex);
        }
        
        return this;
    }

}