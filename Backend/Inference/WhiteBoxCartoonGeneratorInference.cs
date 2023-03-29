using Backend.Model;
using Backend.PostProcessing;
using OpenCvSharp;
using Size = OpenCvSharp.Size;
using static TorchSharp.torch;

namespace Backend.Inference;

public static class WhiteBoxCartoonGeneratorInference
{

    public static IEnumerable<Mat> InferFrames(this WhiteBoxCartoonGenerator whiteBoxCartoonGenerator, 
        IEnumerable<Mat> bgrInputFrames, Size size, Device device)
    {
        Tensor rgbInputFrames = stack(
            bgrInputFrames
                .AsParallel()
                .AsOrdered()
                .Select(
                    frame => Utility.ImageToTensor(frame.CvtColor(ColorConversionCodes.BGR2RGB), size, device)
                ).ToArray()
        );

        // (frames, h, w, c) -> (frames, c, h, w)
        rgbInputFrames = rgbInputFrames.permute(0, 3, 1, 2);
        
        Tensor rgbOutputFrames = whiteBoxCartoonGenerator.forward(rgbInputFrames);
        rgbOutputFrames = tanh(rgbOutputFrames);
        rgbOutputFrames = WhiteBoxCartoonGeneratorPostProcessing.PostProcess(rgbInputFrames, rgbOutputFrames);
        rgbOutputFrames = rgbOutputFrames * 0.5 + 0.5;
        
        // (frames, c, h, w) -> (frames, h, w, c)
        rgbOutputFrames = rgbOutputFrames.permute(0, 2, 3, 1);
        
        // float32 -> uint8
        rgbOutputFrames *= 255;
        rgbOutputFrames = rgbOutputFrames.to_type(ScalarType.Byte);
        
        // device -> cpu
        rgbOutputFrames = rgbOutputFrames.detach().to("cpu");

        // [h, w]
        int[] matSize = { (int)rgbOutputFrames.shape[1], (int)rgbOutputFrames.shape[2] };

        // x: (h, w, c)
        return rgbOutputFrames.unbind()
            .AsParallel()
            .AsOrdered()
            .Select(x =>
            {
                byte[] data = x.flatten().bytes.ToArray();
                return new Mat(matSize, MatType.CV_8UC3, data).CvtColor(ColorConversionCodes.RGB2BGR);
            })
            .ToArray();
    }

}