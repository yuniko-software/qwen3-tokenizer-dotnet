using Microsoft.ML.Tokenizers;

namespace Yuniko.Software.Qwen3Tokenizer;

/// <summary>
/// Tokenizer for Qwen3 models using byte-level BPE (Byte-Pair Encoding).
/// Supports all Qwen3 model variants including LLM, Embedding, Reranker, and Vision-Language models.
/// </summary>
public sealed class Qwen3Tokenizer
{
    private readonly BpeTokenizer _tokenizer;
    private readonly Dictionary<string, int> _addedTokens;
    private readonly int _padTokenId;
    private readonly bool _isForEmbeddingModel;

    /// <summary>
    /// Gets the vocabulary size including added tokens.
    /// </summary>
    public int VocabularySize => _tokenizer.Vocabulary.Count + _addedTokens.Count;

    /// <summary>
    /// Gets the vocabulary dictionary including added tokens.
    /// </summary>
    public IReadOnlyDictionary<string, int> Vocabulary
    {
        get
        {
            var combined = new Dictionary<string, int>(_tokenizer.Vocabulary);
            foreach (var (token, id) in _addedTokens)
            {
                combined[token] = id;
            }
            return combined;
        }
    }

    /// <summary>
    /// Gets all added tokens.
    /// These are treated as atomic during pre-tokenization.
    /// </summary>
    public IReadOnlyDictionary<string, int> AddedTokens => _addedTokens;

    /// <summary>
    /// Gets the token IDs marked as "special": true in HuggingFace.
    /// These are skipped during decoding when skipSpecialTokens=true.
    /// </summary>
    public IReadOnlySet<int> SpecialTokenIds { get; }

    /// <summary>
    /// Creates a Qwen3 tokenizer from vocabulary and merges files.
    /// </summary>
    /// <param name="vocabPath">Path to vocab.json file.</param>
    /// <param name="mergesPath">Path to merges.txt file.</param>
    /// <param name="options">Tokenizer configuration options.</param>
    /// <param name="isForEmbeddingModel">Set to true if tokenizer is created for an embedding model. This will add a pad token to the end of the sequence during encoding.</param>
    private Qwen3Tokenizer(
        string vocabPath,
        string mergesPath,
        Qwen3TokenizerOptions options,
        bool isForEmbeddingModel)
    {
        if (!File.Exists(vocabPath))
        {
            throw new FileNotFoundException($"Vocabulary file not found: {vocabPath}");
        }

        if (!File.Exists(mergesPath))
        {
            throw new FileNotFoundException($"Merges file not found: {mergesPath}");
        }

        _addedTokens = new Dictionary<string, int>(options.AddedTokens);
        SpecialTokenIds = options.SpecialTokenIds;
        _padTokenId = options.PadTokenId;
        _isForEmbeddingModel = isForEmbeddingModel;

        var bpeOptions = new BpeOptions(vocabPath, mergesPath)
        {
            ByteLevel = options.ByteLevel,
            Normalizer = options.Normalizer,
            PreTokenizer = new RegexPreTokenizer(options.PreTokenizerRegex, specialTokens: _addedTokens),
            SpecialTokens = _addedTokens,
        };

        _tokenizer = BpeTokenizer.Create(bpeOptions);
    }

    /// <summary>
    /// Creates a Qwen3 tokenizer from local vocabulary and merges files.
    /// </summary>
    /// <param name="vocabPath">Path to vocab.json file.</param>
    /// <param name="mergesPath">Path to merges.txt file.</param>
    /// <param name="isForEmbeddingModel">Set to true if tokenizer is created for an embedding model. This will add a pad token to the end of the sequence during encoding.</param>
    /// <param name="options">Tokenizer configuration options. If null, uses Qwen3TokenizerOptions.Default.</param>
    /// <returns>A new Qwen3Tokenizer instance.</returns>
    public static Qwen3Tokenizer FromFiles(
        string vocabPath,
        string mergesPath,
        bool isForEmbeddingModel,
        Qwen3TokenizerOptions? options = null)
    {
        return new Qwen3Tokenizer(vocabPath, mergesPath, options ?? Qwen3TokenizerOptions.Default, isForEmbeddingModel);
    }

    /// <summary>
    /// Downloads tokenizer files from HuggingFace and creates a Qwen3 tokenizer.
    /// </summary>
    /// <param name="modelName">Model name (e.g., "Qwen/Qwen3-0.6B", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-VL-30B-A3B-Instruct").</param>
    /// <param name="isForEmbeddingModel">Set to true if tokenizer is created for an embedding model. This will add a pad token to the end of the sequence during encoding.</param>
    /// <param name="cacheDir">Directory to cache downloaded files. If null, uses temporary directory.</param>
    /// <param name="options">Tokenizer configuration options. If null, uses Qwen3TokenizerOptions.Default.</param>
    /// <param name="httpClient">HttpClient to use for downloads. If null, creates a new one.</param>
    /// <returns>A new Qwen3Tokenizer instance.</returns>
    /// <exception cref="ArgumentException">Thrown when model name does not contain 'qwen3' (case-insensitive).</exception>
    public static Qwen3Tokenizer FromHuggingFace(
        string modelName,
        bool isForEmbeddingModel,
        string? cacheDir = null,
        Qwen3TokenizerOptions? options = null,
        HttpClient? httpClient = null)
    {
        ValidateQwen3ModelName(modelName);
        var provider = new HuggingFaceFileProvider(modelName, cacheDir, httpClient);
        var (vocabPath, mergesPath) = provider.GetFiles();
        return new Qwen3Tokenizer(vocabPath, mergesPath, options ?? Qwen3TokenizerOptions.Default, isForEmbeddingModel);
    }

    /// <summary>
    /// Creates a Qwen3 tokenizer using a custom file provider.
    /// </summary>
    /// <param name="fileProvider">The file provider to use for obtaining tokenizer files.</param>
    /// <param name="isForEmbeddingModel">Set to true if tokenizer is created for an embedding model. This will add a pad token to the end of the sequence during encoding.</param>
    /// <param name="options">Tokenizer configuration options. If null, uses Qwen3TokenizerOptions.Default.</param>
    /// <returns>A new Qwen3Tokenizer instance.</returns>
    public static Qwen3Tokenizer FromProvider(
        ITokenizerFileProvider fileProvider,
        bool isForEmbeddingModel,
        Qwen3TokenizerOptions? options = null)
    {
        var (vocabPath, mergesPath) = fileProvider.GetFiles();
        return new Qwen3Tokenizer(vocabPath, mergesPath, options ?? Qwen3TokenizerOptions.Default, isForEmbeddingModel);
    }

    /// <summary>
    /// Asynchronously downloads tokenizer files from HuggingFace and creates a Qwen3 tokenizer.
    /// </summary>
    /// <param name="modelName">Model name (e.g., "Qwen/Qwen3-0.6B", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-VL-30B-A3B-Instruct").</param>
    /// <param name="isForEmbeddingModel">Set to true if tokenizer is created for an embedding model. This will add a pad token to the end of the sequence during encoding.</param>
    /// <param name="cacheDir">Directory to cache downloaded files. If null, uses temporary directory.</param>
    /// <param name="options">Tokenizer configuration options. If null, uses Qwen3TokenizerOptions.Default.</param>
    /// <param name="httpClient">HttpClient to use for downloads. If null, creates a new one.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A new Qwen3Tokenizer instance.</returns>
    /// <exception cref="ArgumentException">Thrown when model name does not contain 'qwen3' (case-insensitive).</exception>
    public static async Task<Qwen3Tokenizer> FromHuggingFaceAsync(
        string modelName,
        bool isForEmbeddingModel,
        string? cacheDir = null,
        Qwen3TokenizerOptions? options = null,
        HttpClient? httpClient = null,
        CancellationToken cancellationToken = default)
    {
        ValidateQwen3ModelName(modelName);
        var provider = new HuggingFaceFileProvider(modelName, cacheDir, httpClient);
        var (vocabPath, mergesPath) = await provider.GetFilesAsync(cancellationToken).ConfigureAwait(false);
        return new Qwen3Tokenizer(vocabPath, mergesPath, options ?? Qwen3TokenizerOptions.Default, isForEmbeddingModel);
    }

    /// <summary>
    /// Asynchronously creates a Qwen3 tokenizer using a custom file provider.
    /// </summary>
    /// <param name="fileProvider">The file provider to use for obtaining tokenizer files.</param>
    /// <param name="isForEmbeddingModel">Set to true if tokenizer is created for an embedding model. This will add a pad token to the end of the sequence during encoding.</param>
    /// <param name="options">Tokenizer configuration options. If null, uses Qwen3TokenizerOptions.Default.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A new Qwen3Tokenizer instance.</returns>
    public static async Task<Qwen3Tokenizer> FromProviderAsync(
        ITokenizerFileProvider fileProvider,
        bool isForEmbeddingModel,
        Qwen3TokenizerOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var (vocabPath, mergesPath) = await fileProvider.GetFilesAsync(cancellationToken).ConfigureAwait(false);
        return new Qwen3Tokenizer(vocabPath, mergesPath, options ?? Qwen3TokenizerOptions.Default, isForEmbeddingModel);
    }

    /// <summary>
    /// Encodes text into token IDs.
    /// </summary>
    /// <param name="text">Input text to tokenize.</param>
    /// <param name="addSpecialTokens">Whether to add special tokens. For embedding models, this adds a pad token at the end. Default is true to match HuggingFace behavior.</param>
    /// <returns>Array of token IDs.</returns>
    public int[] Encode(string text, bool addSpecialTokens = true)
    {
        IReadOnlyList<int> ids = _tokenizer.EncodeToIds(text);

        if (addSpecialTokens && _isForEmbeddingModel)
        {
            var result = new int[ids.Count + 1];
            for (int i = 0; i < ids.Count; i++)
            {
                result[i] = ids[i];
            }
            result[ids.Count] = _padTokenId;
            return result;
        }

        if (ids is int[] array)
        {
            return array;
        }

        var idsArray = new int[ids.Count];
        for (int i = 0; i < ids.Count; i++)
        {
            idsArray[i] = ids[i];
        }

        return idsArray;
    }

    /// <summary>
    /// Encodes text and returns detailed encoding information.
    /// </summary>
    /// <param name="text">Input text to tokenize.</param>
    /// <param name="addSpecialTokens">Whether to add special tokens. For embedding models, this adds a pad token at the end. Default is true to match HuggingFace behavior.</param>
    /// <returns>Detailed encoding result with token IDs, strings, and offsets (in UTF-16 char indices).</returns>
    /// <remarks>
    /// Offsets are returned as UTF-16 char indices (matching C# string indexing).
    /// Emojis and other characters outside the Basic Multilingual Plane use 2 UTF-16 code units (surrogate pairs).
    /// </remarks>
    public EncodingResult EncodeDetailed(string text, bool addSpecialTokens = true)
    {
        IReadOnlyList<EncodedToken> encodedTokens = _tokenizer.EncodeToTokens(text, out string? normalizedText);

        int extraTokens = (addSpecialTokens && _isForEmbeddingModel) ? 1 : 0;
        var ids = new int[encodedTokens.Count + extraTokens];
        var tokens = new string[encodedTokens.Count + extraTokens];
        var offsets = new (int Index, int Length)[encodedTokens.Count + extraTokens];

        for (int i = 0; i < encodedTokens.Count; i++)
        {
            ids[i] = encodedTokens[i].Id;
            tokens[i] = _tokenizer.Decode([encodedTokens[i].Id]) ?? string.Empty;
            offsets[i] = (encodedTokens[i].Offset.Start.Value, encodedTokens[i].Offset.End.Value - encodedTokens[i].Offset.Start.Value);
        }

        if (addSpecialTokens && _isForEmbeddingModel)
        {
            ids[encodedTokens.Count] = _padTokenId;
            tokens[encodedTokens.Count] = _tokenizer.Decode([_padTokenId]) ?? string.Empty;
            offsets[encodedTokens.Count] = (text.Length, 0);
        }

        return new EncodingResult(ids, tokens, offsets);
    }

    /// <summary>
    /// Counts the number of tokens in the text without full encoding.
    /// </summary>
    /// <param name="text">Input text.</param>
    /// <param name="addSpecialTokens">Whether to add special tokens. For embedding models, this adds a pad token at the end. Default is true to match HuggingFace behavior.</param>
    /// <returns>Token count.</returns>
    public int CountTokens(string text, bool addSpecialTokens = true)
    {
        int count = _tokenizer.CountTokens(text);
        return (addSpecialTokens && _isForEmbeddingModel) ? count + 1 : count;
    }

    /// <summary>
    /// Decodes token IDs back to text.
    /// </summary>
    /// <param name="ids">Token IDs to decode.</param>
    /// <param name="skipSpecialTokens">Whether to skip special tokens in output (only skips tokens marked as "special": true).</param>
    /// <returns>Decoded text.</returns>
    public string Decode(int[] ids, bool skipSpecialTokens = true)
    {
        if (skipSpecialTokens)
        {
            ids = [.. ids.Where(id => !SpecialTokenIds.Contains(id))];
        }

        return _tokenizer.Decode(ids) ?? string.Empty;
    }

    /// <summary>
    /// Prepares inputs for ONNX Runtime inference with Qwen3 models.
    /// </summary>
    /// <param name="text">Input text to encode.</param>
    /// <param name="addSpecialTokens">Whether to add special tokens. For embedding models, this adds a pad token at the end. Default is true to match HuggingFace behavior.</param>
    /// <returns>ONNX inputs containing input_ids, attention_mask, and position_ids.</returns>
    /// <remarks>
    /// Returns 1D arrays with dynamic length (no padding) for single-text inference.
    /// Position IDs are sequential (0, 1, 2, ...) for each token in the sequence.
    /// Some models (e.g., embedding models) may not require position_ids.
    /// For batch inference, call this method for each text and construct 2D arrays manually with appropriate padding.
    /// </remarks>
    public OnnxInputs PrepareForOnnx(string text, bool addSpecialTokens = true)
    {
        var ids = Encode(text, addSpecialTokens);

        var inputIds = new long[ids.Length];
        var attentionMask = new long[ids.Length];
        var positionIds = new long[ids.Length];

        for (int i = 0; i < ids.Length; i++)
        {
            inputIds[i] = ids[i];
            attentionMask[i] = 1;
            positionIds[i] = i;
        }

        return new OnnxInputs(inputIds, attentionMask, positionIds, SequenceLength: ids.Length);
    }

    /// <summary>
    /// Validates that the model name contains 'qwen3' (case-insensitive).
    /// </summary>
    /// <param name="modelName">The model name to validate.</param>
    /// <exception cref="ArgumentException">Thrown when model name does not contain 'qwen3'.</exception>
    private static void ValidateQwen3ModelName(string modelName)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));
        }

        if (!modelName.Contains("qwen3", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                $"Model name '{modelName}' does not appear to be a Qwen3 model. " +
                "Expected model name to contain 'qwen3' (e.g., 'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-Embedding-8B').",
                nameof(modelName));
        }
    }
}
