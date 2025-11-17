namespace Yuniko.Software.Qwen3Tokenizer.Tests;

/// <summary>
/// Tests for embedding model-specific functionality, including the pad token behavior
/// when isForEmbeddingModel is set to true.
/// </summary>
public class EmbeddingModelTests
{
    private readonly Qwen3Tokenizer _embeddingTokenizer;
    private readonly Qwen3Tokenizer _nonEmbeddingTokenizer;
    private const string TestText = "Hello world";

    public EmbeddingModelTests()
    {
        _embeddingTokenizer = Qwen3Tokenizer.FromHuggingFace("Qwen/Qwen3-Embedding-0.6B", isForEmbeddingModel: true);
        _nonEmbeddingTokenizer = Qwen3Tokenizer.FromHuggingFace("Qwen/Qwen3-0.6B", isForEmbeddingModel: false);
    }

    [Fact]
    public void Encode_EmbeddingModel_WithSpecialTokens_AddsPadToken()
    {
        var result = _embeddingTokenizer.Encode(TestText, addSpecialTokens: true);
        var resultWithoutSpecial = _embeddingTokenizer.Encode(TestText, addSpecialTokens: false);

        Assert.Equal(resultWithoutSpecial.Length + 1, result.Length);
        Assert.Equal(151643, result[^1]);
    }

    [Fact]
    public void Encode_EmbeddingModel_WithoutSpecialTokens_DoesNotAddPadToken()
    {
        var result = _embeddingTokenizer.Encode(TestText, addSpecialTokens: false);

        Assert.NotEqual(151643, result[^1]);
    }

    [Fact]
    public void Encode_NonEmbeddingModel_WithSpecialTokens_DoesNotAddPadToken()
    {
        var embeddingResult = _embeddingTokenizer.Encode(TestText, addSpecialTokens: true);
        var nonEmbeddingResult = _nonEmbeddingTokenizer.Encode(TestText, addSpecialTokens: true);

        Assert.Equal(nonEmbeddingResult.Length + 1, embeddingResult.Length);
    }

    [Fact]
    public void Encode_NonEmbeddingModel_SpecialTokensParameterDoesNotAffectLength()
    {
        var withSpecial = _nonEmbeddingTokenizer.Encode(TestText, addSpecialTokens: true);
        var withoutSpecial = _nonEmbeddingTokenizer.Encode(TestText, addSpecialTokens: false);

        Assert.Equal(withSpecial.Length, withoutSpecial.Length);
    }

    [Fact]
    public void CountTokens_EmbeddingModel_WithSpecialTokens_IncludesPadToken()
    {
        var countWithSpecial = _embeddingTokenizer.CountTokens(TestText, addSpecialTokens: true);
        var countWithoutSpecial = _embeddingTokenizer.CountTokens(TestText, addSpecialTokens: false);

        Assert.Equal(countWithoutSpecial + 1, countWithSpecial);
    }

    [Fact]
    public void CountTokens_NonEmbeddingModel_SpecialTokensDoesNotAffectCount()
    {
        var countWithSpecial = _nonEmbeddingTokenizer.CountTokens(TestText, addSpecialTokens: true);
        var countWithoutSpecial = _nonEmbeddingTokenizer.CountTokens(TestText, addSpecialTokens: false);

        Assert.Equal(countWithSpecial, countWithoutSpecial);
    }

    [Fact]
    public void EncodeDetailed_EmbeddingModel_WithSpecialTokens_AddsPadToken()
    {
        var result = _embeddingTokenizer.EncodeDetailed(TestText, addSpecialTokens: true);
        var resultWithoutSpecial = _embeddingTokenizer.EncodeDetailed(TestText, addSpecialTokens: false);

        Assert.Equal(resultWithoutSpecial.Ids.Length + 1, result.Ids.Length);
        Assert.Equal(resultWithoutSpecial.Tokens.Length + 1, result.Tokens.Length);
        Assert.Equal(resultWithoutSpecial.Offsets.Length + 1, result.Offsets.Length);

        Assert.Equal(151643, result.Ids[^1]);
        Assert.Equal("<|endoftext|>", result.Tokens[^1]);

        Assert.Equal(TestText.Length, result.Offsets[^1].Index);
        Assert.Equal(0, result.Offsets[^1].Length);
    }

    [Fact]
    public void EncodeDetailed_EmbeddingModel_WithoutSpecialTokens_DoesNotAddPadToken()
    {
        var result = _embeddingTokenizer.EncodeDetailed(TestText, addSpecialTokens: false);

        Assert.NotEqual(151643, result.Ids[^1]);
    }

    [Fact]
    public void EncodeDetailed_NonEmbeddingModel_SpecialTokensDoesNotAffectLength()
    {
        var withSpecial = _nonEmbeddingTokenizer.EncodeDetailed(TestText, addSpecialTokens: true);
        var withoutSpecial = _nonEmbeddingTokenizer.EncodeDetailed(TestText, addSpecialTokens: false);

        Assert.Equal(withSpecial.Ids.Length, withoutSpecial.Ids.Length);
        Assert.Equal(withSpecial.Tokens.Length, withoutSpecial.Tokens.Length);
    }

    [Theory]
    [InlineData("")]
    [InlineData("A")]
    [InlineData("This is a longer text that should still get the pad token added at the end")]
    public void Encode_EmbeddingModel_VariousLengths_AlwaysAddsPadTokenWhenEnabled(string text)
    {
        var withSpecial = _embeddingTokenizer.Encode(text, addSpecialTokens: true);
        var withoutSpecial = _embeddingTokenizer.Encode(text, addSpecialTokens: false);

        if (string.IsNullOrEmpty(text))
        {
            Assert.Single(withSpecial);
            Assert.Empty(withoutSpecial);
            Assert.Equal(151643, withSpecial[0]);
        }
        else
        {
            Assert.Equal(withoutSpecial.Length + 1, withSpecial.Length);
            Assert.Equal(151643, withSpecial[^1]);
        }
    }

    [Fact]
    public void Decode_EmbeddingModel_PadToken_SkippedWhenRequested()
    {
        var encoded = _embeddingTokenizer.Encode(TestText, addSpecialTokens: true);

        var decodedWithSpecial = _embeddingTokenizer.Decode(encoded, skipSpecialTokens: false);
        var decodedSkipSpecial = _embeddingTokenizer.Decode(encoded, skipSpecialTokens: true);

        Assert.Contains("<|endoftext|>", decodedWithSpecial, StringComparison.Ordinal);
        Assert.DoesNotContain("<|endoftext|>", decodedSkipSpecial, StringComparison.Ordinal);
    }

    [Fact]
    public void PrepareForOnnx_EmbeddingModel_WithSpecialTokens_IncludesPadToken()
    {
        var result = _embeddingTokenizer.PrepareForOnnx(TestText, addSpecialTokens: true);
        var encoded = _embeddingTokenizer.Encode(TestText, addSpecialTokens: true);

        Assert.Equal(encoded.Length, result.InputIds.Length);
        for (int i = 0; i < encoded.Length; i++)
        {
            Assert.Equal(encoded[i], result.InputIds[i]);
            Assert.Equal(1L, result.AttentionMask[i]);
        }
    }

    [Fact]
    public void PrepareForOnnx_EmbeddingModel_WithoutSpecialTokens_DoesNotIncludePadToken()
    {
        var resultWithSpecial = _embeddingTokenizer.PrepareForOnnx(TestText, addSpecialTokens: true);
        var resultWithoutSpecial = _embeddingTokenizer.PrepareForOnnx(TestText, addSpecialTokens: false);

        var encodedWith = _embeddingTokenizer.Encode(TestText, addSpecialTokens: true);
        var encodedWithout = _embeddingTokenizer.Encode(TestText, addSpecialTokens: false);

        Assert.Equal(encodedWith.Length, resultWithSpecial.InputIds.Length);
        Assert.Equal(encodedWithout.Length, resultWithoutSpecial.InputIds.Length);

        for (int i = 0; i < encodedWith.Length; i++)
        {
            Assert.Equal(encodedWith[i], resultWithSpecial.InputIds[i]);
        }

        for (int i = 0; i < encodedWithout.Length; i++)
        {
            Assert.Equal(encodedWithout[i], resultWithoutSpecial.InputIds[i]);
        }
    }

    [Fact]
    public void PrepareForOnnx_NonEmbeddingModel_SpecialTokensParameterDoesNotAffectEncoding()
    {
        var withSpecial = _nonEmbeddingTokenizer.PrepareForOnnx(TestText, addSpecialTokens: true);
        var withoutSpecial = _nonEmbeddingTokenizer.PrepareForOnnx(TestText, addSpecialTokens: false);

        Assert.Equal(withSpecial.InputIds, withoutSpecial.InputIds);
        Assert.Equal(withSpecial.AttentionMask, withoutSpecial.AttentionMask);
    }
}
