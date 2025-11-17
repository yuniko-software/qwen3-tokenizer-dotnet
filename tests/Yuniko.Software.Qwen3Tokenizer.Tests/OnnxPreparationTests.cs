namespace Yuniko.Software.Qwen3Tokenizer.Tests;

public class OnnxPreparationTests
{
    private readonly Qwen3Tokenizer _tokenizer;

    public OnnxPreparationTests()
    {
        _tokenizer = Qwen3Tokenizer.FromHuggingFace("Qwen/Qwen3-0.6B", isForEmbeddingModel: false);
    }

    [Fact]
    public void PrepareForOnnx_UsesActualTokenCount()
    {
        const string text = "Hello world";

        var result = _tokenizer.PrepareForOnnx(text);
        var tokenCount = _tokenizer.CountTokens(text, addSpecialTokens: true);

        Assert.Equal(tokenCount, result.InputIds.Length);
        Assert.Equal(tokenCount, result.AttentionMask.Length);
        Assert.Equal(tokenCount, result.PositionIds.Length);
        Assert.Equal(tokenCount, result.SequenceLength);
    }

    [Fact]
    public void PrepareForOnnx_NoPadding()
    {
        const string text = "Short";

        var result = _tokenizer.PrepareForOnnx(text);

        Assert.All(result.AttentionMask, mask => Assert.Equal(1L, mask));
    }

    [Theory]
    [InlineData("Hi")]
    [InlineData("This is a test")]
    [InlineData("A much longer sentence with more tokens to encode")]
    public void PrepareForOnnx_MatchesTokenCount(string text)
    {
        var result = _tokenizer.PrepareForOnnx(text);
        var tokenCount = _tokenizer.CountTokens(text, addSpecialTokens: true);

        Assert.Equal(tokenCount, result.SequenceLength);
        Assert.All(result.AttentionMask, mask => Assert.Equal(1L, mask));
    }

    [Fact]
    public void PrepareForOnnx_CreatesSequentialPositionIds()
    {
        const string text = "Hello world test";

        var result = _tokenizer.PrepareForOnnx(text);

        for (int i = 0; i < result.PositionIds.Length; i++)
        {
            Assert.Equal(i, result.PositionIds[i]);
        }
    }

    [Fact]
    public void PrepareForOnnx_EncodesCorrectly()
    {
        const string text = "How are you?";

        var result = _tokenizer.PrepareForOnnx(text);
        var encoded = _tokenizer.Encode(text, addSpecialTokens: true);

        Assert.Equal(encoded.Length, result.InputIds.Length);
        for (int i = 0; i < encoded.Length; i++)
        {
            Assert.Equal(encoded[i], result.InputIds[i]);
        }
    }
}
