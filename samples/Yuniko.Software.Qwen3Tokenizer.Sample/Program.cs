using Yuniko.Software.Qwen3Tokenizer;

Console.WriteLine("=== Qwen3 Tokenizer Sample ===\n");

// ============================================================================
// 1. Load Tokenizer
// ============================================================================
Console.WriteLine("1. Loading Tokenizer");
Console.WriteLine("-------------------");

var tokenizer = Qwen3Tokenizer.FromHuggingFace("Qwen/Qwen3-0.6B", isForEmbeddingModel: false);
Console.WriteLine("Loaded tokenizer for Qwen/Qwen3-0.6B");
Console.WriteLine($"Vocabulary size: {tokenizer.VocabularySize:N0}");
Console.WriteLine($"Added tokens: {tokenizer.AddedTokens.Count}");
Console.WriteLine($"Special tokens: {tokenizer.SpecialTokenIds.Count}\n");

// ============================================================================
// 2. Basic Encoding & Decoding
// ============================================================================
Console.WriteLine("2. Basic Encoding & Decoding");
Console.WriteLine("-----------------------------");

string text = "Hello, this is a test of the Qwen3 tokenizer!";
Console.WriteLine($"Text: \"{text}\"");

// Encode text into token IDs
int[] tokenIds = tokenizer.Encode(text);
Console.WriteLine($"Token IDs: [{string.Join(", ", tokenIds)}]");
Console.WriteLine($"Token count: {tokenIds.Length}");

// Decode token IDs back to text
string decoded = tokenizer.Decode(tokenIds);
Console.WriteLine($"Decoded: \"{decoded}\"");

// Count tokens without full encoding (faster)
int count = tokenizer.CountTokens(text);
Console.WriteLine($"Token count (fast): {count}\n");

// ============================================================================
// 3. Detailed Encoding Information
// ============================================================================
Console.WriteLine("3. Detailed Encoding Information");
Console.WriteLine("--------------------------------");

string detailedText = "Hello, world!";
var result = tokenizer.EncodeDetailed(detailedText);

Console.WriteLine($"Text: \"{detailedText}\"");
Console.WriteLine("Token breakdown:");
for (int i = 0; i < result.Ids.Length; i++)
{
    Console.WriteLine($"  [{i}] ID: {result.Ids[i],6} | Token: \"{result.Tokens[i]}\" | " +
                     $"Offset: {result.Offsets[i].Index}, Length: {result.Offsets[i].Length}");
}
Console.WriteLine();

// ============================================================================
// 4. Working with Special Tokens
// ============================================================================
Console.WriteLine("4. Working with Special Tokens");
Console.WriteLine("-------------------------------");

string chatText = "<|im_start|>user\nHello!<|im_end|>";
int[] chatIds = tokenizer.Encode(chatText);

Console.WriteLine($"Original: \"{chatText}\"");
Console.WriteLine($"Token IDs: [{string.Join(", ", chatIds)}]");

string withSpecial = tokenizer.Decode(chatIds, skipSpecialTokens: false);
string withoutSpecial = tokenizer.Decode(chatIds, skipSpecialTokens: true);

Console.WriteLine($"Decoded (with special tokens): \"{withSpecial}\"");
Console.WriteLine($"Decoded (without special tokens): \"{withoutSpecial}\"");

// Access special token constants
Console.WriteLine("\nSpecial token constants:");
Console.WriteLine($"  EOS token: '{Qwen3Tokens.ImEnd}' (ID: {Qwen3Tokens.ImEndTokenId})");
Console.WriteLine($"  PAD token: '{Qwen3Tokens.EndOfText}' (ID: {Qwen3Tokens.EndOfTextTokenId})");
Console.WriteLine($"  Start token: '{Qwen3Tokens.ImStart}' (ID: {Qwen3Tokens.ImStartTokenId})\n");

// ============================================================================
// 5. ONNX Runtime Integration
// ============================================================================
Console.WriteLine("5. ONNX Runtime Integration");
Console.WriteLine("---------------------------");

string onnxText = "How are you today?";

var onnxInputs = tokenizer.PrepareForOnnx(onnxText);
Console.WriteLine($"Text: \"{onnxText}\"");
Console.WriteLine($"Sequence length: {onnxInputs.SequenceLength}");
Console.WriteLine($"Input IDs: [{string.Join(", ", onnxInputs.InputIds)}]");
Console.WriteLine($"Attention mask: [{string.Join(", ", onnxInputs.AttentionMask)}]");
Console.WriteLine($"Position IDs: [{string.Join(", ", onnxInputs.PositionIds)}]");
Console.WriteLine("Note: Some models (e.g., embedding models) may not require position_ids\n");

// ============================================================================
// 6. Encoding with Special Tokens
// ============================================================================
Console.WriteLine("6. Encoding with Special Tokens");
Console.WriteLine("--------------------------------");

string specialText = "This is a sentence.";
int[] withoutSpecialTokens = tokenizer.Encode(specialText, addSpecialTokens: false);
int[] withSpecialTokens = tokenizer.Encode(specialText, addSpecialTokens: true);

Console.WriteLine($"Text: \"{specialText}\"");
Console.WriteLine($"Without special tokens: [{string.Join(", ", withoutSpecialTokens)}] (length: {withoutSpecialTokens.Length})");
Console.WriteLine($"With special tokens: [{string.Join(", ", withSpecialTokens)}] (length: {withSpecialTokens.Length})");
Console.WriteLine("Note: For non-embedding models, special tokens don't add extra tokens by default\n");

// ============================================================================
// 7. Embedding Model Example
// ============================================================================
Console.WriteLine("7. Embedding Model Example");
Console.WriteLine("--------------------------");

var embeddingTokenizer = Qwen3Tokenizer.FromHuggingFace("Qwen/Qwen3-Embedding-0.6B", isForEmbeddingModel: true);
Console.WriteLine("Loaded embedding tokenizer for Qwen/Qwen3-Embedding-0.6B");

string embText = "This is a test sentence for embeddings.";
int[] embWithoutSpecial = embeddingTokenizer.Encode(embText, addSpecialTokens: false);
int[] embWithSpecial = embeddingTokenizer.Encode(embText, addSpecialTokens: true);

Console.WriteLine($"Text: \"{embText}\"");
Console.WriteLine($"Without special tokens: [{string.Join(", ", embWithoutSpecial)}] (length: {embWithoutSpecial.Length})");
Console.WriteLine($"With special tokens: [{string.Join(", ", embWithSpecial)}] (length: {embWithSpecial.Length})");
Console.WriteLine("Note: Embedding models add a pad token at the end when addSpecialTokens=true");
Console.WriteLine($"Pad token ID: {Qwen3Tokens.EndOfTextTokenId}\n");

// ============================================================================
// 8. Multilingual and Emoji Support
// ============================================================================
Console.WriteLine("8. Multilingual and Emoji Support");
Console.WriteLine("---------------------------------");

string[] examples =
[
    "Hello, world!",
    "你好，世界！",
    "こんにちは世界",
    "Привет мир",
    "Hello 👋 world 🌍",
];

foreach (string example in examples)
{
    int[] ids = tokenizer.Encode(example);
    Console.WriteLine($"Text: \"{example}\"");
    Console.WriteLine($"  Tokens: {ids.Length}");
}

Console.WriteLine("\n=== Sample Complete ===");
