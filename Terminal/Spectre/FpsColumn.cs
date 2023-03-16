using Spectre.Console;
using Spectre.Console.Rendering;

namespace Terminal.Spectre;

public sealed class FpsColumn : ProgressColumn
{
    
    private readonly Style Style = new(foreground: Color.Orange1);

    public int BatchSize { private get; init; } = 1;

    public override IRenderable Render(RenderOptions options, ProgressTask task, TimeSpan deltaTime)
    {
        double? speed = task.Speed;
        if (speed is null) return new Markup("NA", Style);

        double actualSpeed = speed.Value * BatchSize;
        return new Markup("FPS: " + actualSpeed.ToString("F1"), Style);
    }

}