# Yuniko.Software.Qwen3Tokenizer

[![CI](https://github.com/yuniko-software/qwen3-tokenizer/actions/workflows/ci.yml/badge.svg)](https://github.com/yuniko-software/qwen3-tokenizer/actions/workflows/ci.yml)
[![NuGet](https://img.shields.io/nuget/v/Yuniko.Software.Qwen3Tokenizer.svg)](https://www.nuget.org/packages/Yuniko.Software.Qwen3Tokenizer)
[![NuGet Downloads](https://img.shields.io/nuget/dt/Yuniko.Software.Qwen3Tokenizer.svg)](https://www.nuget.org/packages/Yuniko.Software.Qwen3Tokenizer)

Native .NET tokenizer implementation for Qwen3 models. Lightweight byte-pair encoding with HuggingFace integration.

## Features

- **Native .NET Implementation**: Qwen3 tokenizer built specifically for .NET applications
- **Identical to HuggingFace**: Produces the same token IDs and outputs as the official HuggingFace tokenizer
- **Qwen Model Family Support**: Compatible with all Qwen3 variants (LLM, Embedding, Reranker, and Vision-Language models)
- **HuggingFace Integration**: Load tokenizer files directly from HuggingFace model repositories
- **Configurable**: Customize tokenization behavior through options and custom file providers
- **No External Dependencies**: Does not require Python or other runtime dependencies

## Installation

```bash
dotnet add package Yuniko.Software.Qwen3Tokenizer
```

Or via Package Manager:

```powershell
Install-Package Yuniko.Software.Qwen3Tokenizer
```

## Quick Start

```csharp
using Yuniko.Software.Qwen3Tokenizer;

// Load from HuggingFace model (specify if it's for an embedding model)
var tokenizer = await Qwen3Tokenizer.FromHuggingFaceAsync(
    "Qwen/Qwen3-0.6B", 
    isForEmbeddingModel: false
);

// Encode text
int[] tokenIds = tokenizer.Encode("Hello, world!");
Console.WriteLine($"Token IDs: [{string.Join(", ", tokenIds)}]");
Console.WriteLine($"Token count: {tokenIds.Length}");

// Decode tokens
string decodedText = tokenizer.Decode(tokenIds);
Console.WriteLine($"Decoded: {decodedText}");
```

## Usage Examples

### Basic Tokenization

```csharp
// For regular LLM models
var tokenizer = Qwen3Tokenizer.FromHuggingFace("Qwen/Qwen3-0.6B", isForEmbeddingModel: false);

// Encode text into token IDs
int[] ids = tokenizer.Encode("The quick brown fox jumps over the lazy dog");

// Decode token IDs back to text
string text = tokenizer.Decode(ids);

// Count tokens without full encoding
int tokenCount = tokenizer.CountTokens("Some text to count");
```

### Working with Embedding Models

```csharp
// For embedding models - adds pad token at the end when addSpecialTokens=true
var embeddingTokenizer = Qwen3Tokenizer.FromHuggingFace(
    "Qwen/Qwen3-Embedding-0.6B", 
    isForEmbeddingModel: true
);

// With special tokens (includes pad token at the end)
int[] withSpecial = embeddingTokenizer.Encode("Your text here", addSpecialTokens: true);

// Without special tokens
int[] withoutSpecial = embeddingTokenizer.Encode("Your text here", addSpecialTokens: false);

Console.WriteLine($"With special tokens: {withSpecial.Length} tokens");
Console.WriteLine($"Without special tokens: {withoutSpecial.Length} tokens");
```

### Decoding with Special Tokens

```csharp
// Encode text with special tokens
string chatMessage = "<|im_start|>user\nHello!<|im_end|>";
int[] tokens = tokenizer.Encode(chatMessage);

// Decode with special tokens preserved
string withSpecial = tokenizer.Decode(tokens, skipSpecialTokens: false);

// Decode with special tokens removed (default behavior)
string withoutSpecial = tokenizer.Decode(tokens, skipSpecialTokens: true);

Console.WriteLine($"With special tokens: {withSpecial}");
Console.WriteLine($"Without special tokens: {withoutSpecial}");
```

### Detailed Encoding Information

```csharp
// Get detailed information about tokens
var result = tokenizer.EncodeDetailed("Hello, world!");

for (int i = 0; i < result.Ids.Length; i++)
{
    Console.WriteLine($"Token: '{result.Tokens[i]}' | ID: {result.Ids[i]} | " +
                     $"Offset: {result.Offsets[i].Index}, Length: {result.Offsets[i].Length}");
}
```

### ONNX Runtime Integration

```csharp
// Prepare inputs for ONNX Runtime inference (dynamic length, no padding)
var inputs = tokenizer.PrepareForOnnx("Your input text here");

// Use with ONNX Runtime
// Note: Some models (e.g., embedding models) may not require position_ids
long[] inputIds = inputs.InputIds;
long[] attentionMask = inputs.AttentionMask;
long[] positionIds = inputs.PositionIds;
```

### Loading from Local Files

```csharp
// Load tokenizer from local vocabulary and merges files
var tokenizer = Qwen3Tokenizer.FromFiles(
    vocabPath: "/path/to/vocab.json",
    mergesPath: "/path/to/merges.txt",
    isForEmbeddingModel: false
);
```

### Custom File Provider

```csharp
// Implement custom file provider for advanced scenarios
public class CustomFileProvider : ITokenizerFileProvider
{
    public (string VocabPath, string MergesPath) GetFiles()
    {
        // Custom logic to provide tokenizer files
        return ("/path/to/vocab.json", "/path/to/merges.txt");
    }

    public Task<(string VocabPath, string MergesPath)> GetFilesAsync(
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(GetFiles());
    }
}

// Use custom provider
var tokenizer = Qwen3Tokenizer.FromProvider(new CustomFileProvider(), isForEmbeddingModel: false);
```

### Accessing Vocabulary and Token Information

```csharp
// Get vocabulary size
int vocabSize = tokenizer.VocabularySize;

// Access full vocabulary
IReadOnlyDictionary<string, int> vocab = tokenizer.Vocabulary;

// Access added tokens (special tokens and others)
IReadOnlyDictionary<string, int> addedTokens = tokenizer.AddedTokens;

// Access special token IDs
IReadOnlySet<int> specialTokenIds = tokenizer.SpecialTokenIds;

// Use predefined token constants
Console.WriteLine($"IM_END token: {Qwen3Tokens.ImEnd} (ID: {Qwen3Tokens.ImEndTokenId})");
Console.WriteLine($"PAD token: {Qwen3Tokens.EndOfText} (ID: {Qwen3Tokens.EndOfTextTokenId})");
Console.WriteLine($"IM_START token: {Qwen3Tokens.ImStart} (ID: {Qwen3Tokens.ImStartTokenId})");
```

For more examples, see the [sample project](samples/Yuniko.Software.Qwen3Tokenizer.Sample).

## Supported Models

Works with all Qwen3 model variants:
- Qwen3 LLM models (text generation)
- Qwen3-Embedding models (text embeddings)
- Qwen3-Reranker models (document reranking)
- Qwen3-VL models (vision-language)

## Requirements

- .NET 10.0 or later

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/yuniko-software/qwen3-tokenizer/blob/main/LICENSE) file for details.

## Contributing

Contributions are welcome! Please visit the [GitHub repository](https://github.com/yuniko-software/qwen3-tokenizer) for more information.

## Support

For issues, questions, or suggestions, please open an issue on [GitHub](https://github.com/yuniko-software/qwen3-tokenizer/issues).

---

⭐ If you find this project useful, please consider giving it a star on GitHub! ⭐

Your support helps make this project more visible to other developers who might benefit from a native .NET implementation of the Qwen3 tokenizer.
