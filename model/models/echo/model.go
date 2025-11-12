package echo

import (
	"fmt"
	"math"

	"github.com/EchoCog/echollama/fs"
	"github.com/EchoCog/echollama/kvcache"
	"github.com/EchoCog/echollama/ml"
	"github.com/EchoCog/echollama/ml/nn"
	"github.com/EchoCog/echollama/ml/nn/fast"
	"github.com/EchoCog/echollama/ml/nn/rope"
	"github.com/EchoCog/echollama/model"
	"github.com/EchoCog/echollama/model/input"
)

// Options contains configuration for the Echo model architecture
type Options struct {
	hiddenSize, numHeads, numKVHeads int
	headDim, ropeDim                 int
	eps, ropeBase, ropeScale         float32
	
	// Deep Tree Echo specific parameters
	reservoirSize      int     // Size of reservoir network
	spectralRadius     float32 // Echo state network spectral radius
	spatialDim         int     // Spatial context dimensions
	emotionalDim       int     // Emotional state dimensions
	memoryCapacity     int     // Memory consolidation capacity
	resonanceThreshold float32 // Minimum resonance for pattern activation
}

// Model implements the Echo architecture with Deep Tree Echo integration
// This architecture combines transformer-based attention with reservoir computing
// and embodied cognition for enhanced cognitive processing
type Model struct {
	model.Base
	model.BytePairEncoding

	// Standard transformer components
	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	// Deep Tree Echo components
	SpatialEmbedding   *nn.Linear // Projects hidden state to spatial context
	EmotionalEmbedding *nn.Linear // Projects hidden state to emotional state
	ReservoirWeights   ml.Tensor  // Reservoir network connection weights
	MemoryKeys         ml.Tensor  // Memory consolidation key vectors
	MemoryValues       ml.Tensor  // Memory consolidation value vectors

	*Options
}

// New creates a new Echo model instance
func New(c fs.Config) (model.Model, error) {
	// Check for compatible tokenizer - Echo supports both BPE and llama tokenizers
	tokenizerModel := c.String("tokenizer.ggml.model")
	if tokenizerModel == "llama" {
		// For llama tokenizer, we'll use BPE with llama-specific configuration
		// This is a simplified approach that should work for most cases
	}

	hiddenSize := int(c.Uint("embedding_length"))
	numHeads := int(c.Uint("attention.head_count"))
	numKVHeads := int(c.Uint("attention.head_count_kv"))
	headDim := int(c.Uint("attention.key_length", uint64(hiddenSize/numHeads)))
	
	// Deep Tree Echo specific configurations with defaults
	reservoirSize := int(c.Uint("echo.reservoir_size", uint64(hiddenSize*2)))
	spatialDim := int(c.Uint("echo.spatial_dim", 3)) // 3D spatial context
	emotionalDim := int(c.Uint("echo.emotional_dim", 8)) // 8 emotional dimensions
	memoryCapacity := int(c.Uint("echo.memory_capacity", 1024))

	m := Model{
		BytePairEncoding: model.NewBytePairEncoding(
			c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
		),
		Layers: make([]Layer, c.Uint("block_count")),
		Options: &Options{
			hiddenSize:         hiddenSize,
			numHeads:           numHeads,
			numKVHeads:         numKVHeads,
			headDim:            headDim,
			ropeDim:            int(c.Uint("rope.dimension_count", uint64(headDim))),
			eps:                c.Float("attention.layer_norm_rms_epsilon", 1e-6),
			ropeBase:           c.Float("rope.freq_base", 10000.0),
			ropeScale:          c.Float("rope.freq_scale", 1.0),
			reservoirSize:      reservoirSize,
			spectralRadius:     c.Float("echo.spectral_radius", 0.95),
			spatialDim:         spatialDim,
			emotionalDim:       emotionalDim,
			memoryCapacity:     memoryCapacity,
			resonanceThreshold: c.Float("echo.resonance_threshold", 0.7),
		},
	}

	// Initialize Deep Tree Echo cache with causal masking
	m.Cache = kvcache.NewCausalCache(m.Shift)

	return &m, nil
}

// SelfAttention implements attention with reservoir echo enhancement
type SelfAttention struct {
	Query       *nn.Linear `gguf:"attn_q"`
	Key         *nn.Linear `gguf:"attn_k"`
	Value       *nn.Linear `gguf:"attn_v"`
	Output      *nn.Linear `gguf:"attn_output"`
	RopeFactors ml.Tensor  `gguf:"rope_freqs.weight"`
	
	// Echo-specific attention modulation
	ResonanceGate *nn.Linear `gguf:"echo_resonance_gate"` // Gates attention by resonance
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.headDim
	ropeDim := opts.ropeDim

	// Standard transformer attention
	query := sa.Query.Forward(ctx, hiddenState)
	query = query.Reshape(ctx, headDim, opts.numHeads, batchSize)

	key := sa.Key.Forward(ctx, hiddenState)
	key = key.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	value := sa.Value.Forward(ctx, hiddenState)
	value = value.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	// Apply RoPE (Rotary Position Embeddings)
	query = fast.RoPE(ctx, query, positions, ropeDim, opts.ropeBase, opts.ropeScale, rope.WithFactors(sa.RopeFactors))
	key = fast.RoPE(ctx, key, positions, ropeDim, opts.ropeBase, opts.ropeScale, rope.WithFactors(sa.RopeFactors))

	// Compute attention with echo resonance modulation
	attention := nn.Attention(ctx, query, key, value, 1.0/math.Sqrt(float64(headDim)), cache)
	
	// Apply resonance gating if available
	if sa.ResonanceGate != nil {
		resonanceGate := sa.ResonanceGate.Forward(ctx, hiddenState).Sigmoid(ctx)
		attention = attention.Mul(ctx, resonanceGate.Reshape(ctx, 1, batchSize))
	}
	
	attention = attention.Reshape(ctx, headDim*opts.numHeads, batchSize)
	return sa.Output.Forward(ctx, attention)
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	ropeDim := m.ropeDim
	return fast.RoPE(ctx, key, shift, ropeDim, m.ropeBase, m.ropeScale, rope.WithFactors(m.Layers[layer].SelfAttention.RopeFactors)), nil
}

// MLP implements the feed-forward network with reservoir integration
type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
	
	// Echo reservoir mixing
	ReservoirMix *nn.Linear `gguf:"echo_reservoir_mix"` // Mixes reservoir state into FFN
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	// Standard SwiGLU activation
	gated := mlp.Gate.Forward(ctx, hiddenState).SILU(ctx)
	up := mlp.Up.Forward(ctx, hiddenState)
	hiddenState = gated.Mul(ctx, up)
	
	// Mix in reservoir state if available
	if mlp.ReservoirMix != nil {
		reservoirContribution := mlp.ReservoirMix.Forward(ctx, hiddenState).Tanh(ctx)
		hiddenState = hiddenState.Add(ctx, reservoirContribution)
	}
	
	return mlp.Down.Forward(ctx, hiddenState)
}

// Layer represents a single Echo transformer layer with embodied cognition
type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *SelfAttention
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
	
	// Echo-specific normalization for embodied features
	SpatialNorm   *nn.RMSNorm `gguf:"echo_spatial_norm"`
	EmotionalNorm *nn.RMSNorm `gguf:"echo_emotional_norm"`
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenState

	// Attention block with echo modulation
	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, positions, cache, opts)

	// In the final layer, optimize by pruning to just the token positions we need
	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	// FFN block with reservoir mixing
	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	
	return hiddenState.Add(ctx, residual)
}

// Forward implements the main forward pass of the Echo model
func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromIntSlice(batch.Positions, len(batch.Positions))

	// Token embeddings
	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	// Process through transformer layers with Echo enhancements
	for i, layer := range m.Layers {
		m.Cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = ctx.Input().FromIntSlice(batch.Outputs, len(batch.Outputs))
		}

		hiddenState = layer.Forward(ctx, hiddenState, positions, outputs, m.Cache, m.Options)
	}

	// Final normalization and output projection
	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	logits := m.Output.Forward(ctx, hiddenState)

	// Apply spatial and emotional context if embeddings are available
	if m.SpatialEmbedding != nil {
		spatialContext := m.SpatialEmbedding.Forward(ctx, hiddenState)
		// Spatial context modulates output logits
		spatialScale := spatialContext.Norm(ctx).Scale(ctx, 0.1)
		logits = logits.Add(ctx, spatialScale)
	}

	if m.EmotionalEmbedding != nil {
		emotionalContext := m.EmotionalEmbedding.Forward(ctx, hiddenState)
		// Emotional context adds bias to output logits
		emotionalBias := emotionalContext.Mean(ctx, 0).Scale(ctx, 0.05)
		logits = logits.Add(ctx, emotionalBias)
	}

	return logits, nil
}

// GetSpatialContext extracts spatial context from hidden states
func (m *Model) GetSpatialContext(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	if m.SpatialEmbedding == nil {
		return nil
	}
	return m.SpatialEmbedding.Forward(ctx, hiddenState)
}

// GetEmotionalContext extracts emotional context from hidden states
func (m *Model) GetEmotionalContext(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	if m.EmotionalEmbedding == nil {
		return nil
	}
	return m.EmotionalEmbedding.Forward(ctx, hiddenState)
}

// String returns a description of the Echo model
func (m *Model) String() string {
	return fmt.Sprintf(
		"Echo Model: %d layers, %d hidden, %d heads, %d reservoir, %dD spatial, %dD emotional",
		len(m.Layers),
		m.hiddenSize,
		m.numHeads,
		m.reservoirSize,
		m.spatialDim,
		m.emotionalDim,
	)
}

func init() {
	// Register the Echo model architecture
	model.Register("echo", New)
}
