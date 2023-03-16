using Spectre.Console;

namespace Terminal.Spectre;

public sealed class MoonSpinner : Spinner
{

    public override TimeSpan Interval { get; } = TimeSpan.FromMilliseconds(320);

    public override bool IsUnicode => true;

    public override IReadOnlyList<string> Frames { get; } = new List<string>
    {
        "🌑 ",
        "🌒 ",
        "🌓 ",
        "🌔 ",
        "🌕 ",
        "🌖 ",
        "🌗 ",
        "🌘 "
    };

}