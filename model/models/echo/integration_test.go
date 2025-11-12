package echo

import (
	"context"
	"testing"

	"github.com/EchoCog/echollama/core/deeptreeecho"
)

// TestCognitiveIntegration tests the cognitive integration layer
func TestCognitiveIntegration(t *testing.T) {
	// Create a minimal Echo model for testing
	config := &mockConfig{
		data: map[string]interface{}{
			"embedding_length":                   512,
			"attention.head_count":              8,
			"attention.head_count_kv":           8,
			"block_count":                       6,
			"tokenizer.ggml.model":              "gpt2",
			"tokenizer.ggml.tokens":             []string{"<pad>", "hello"},
			"tokenizer.ggml.token_type":         []int{0, 1},
			"tokenizer.ggml.merges":             []string{},
			"tokenizer.ggml.add_bos_token":      true,
			"tokenizer.ggml.bos_token_id":       0,
		},
	}

	model, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create Echo model: %v", err)
	}

	echoModel := model.(*Model)

	// Create a Deep Tree Echo identity
	identity := deeptreeecho.NewIdentity("TestIdentity")

	// Create cognitive integration
	integration := NewCognitiveIntegration(echoModel, identity)
	if integration == nil {
		t.Fatal("Expected non-nil cognitive integration")
	}

	// Test pre-processing
	ctx := context.Background()
	input := "Hello, world!"
	enhanced, err := integration.PreProcess(ctx, input)
	if err != nil {
		t.Fatalf("PreProcess failed: %v", err)
	}

	if enhanced == "" {
		t.Error("Expected non-empty enhanced input")
	}

	// Test post-processing
	output := "This is a test response"
	processed, err := integration.PostProcess(ctx, output)
	if err != nil {
		t.Fatalf("PostProcess failed: %v", err)
	}

	if processed == "" {
		t.Error("Expected non-empty processed output")
	}

	// Verify metrics
	metrics := integration.GetMetrics()
	if metrics.InferenceCount == 0 {
		t.Error("Expected inference count > 0")
	}

	if metrics.Coherence < 0 || metrics.Coherence > 1 {
		t.Errorf("Expected coherence in [0,1], got %f", metrics.Coherence)
	}
}

// TestCognitiveIntegrationFeatures tests enabling/disabling features
func TestCognitiveIntegrationFeatures(t *testing.T) {
	config := &mockConfig{
		data: map[string]interface{}{
			"embedding_length":                   512,
			"attention.head_count":              8,
			"attention.head_count_kv":           8,
			"block_count":                       6,
			"tokenizer.ggml.model":              "gpt2",
			"tokenizer.ggml.tokens":             []string{"<pad>"},
			"tokenizer.ggml.token_type":         []int{0},
			"tokenizer.ggml.merges":             []string{},
			"tokenizer.ggml.add_bos_token":      true,
			"tokenizer.ggml.bos_token_id":       0,
		},
	}

	model, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create Echo model: %v", err)
	}

	echoModel := model.(*Model)
	identity := deeptreeecho.NewIdentity("TestIdentity")
	integration := NewCognitiveIntegration(echoModel, identity)

	// Test disabling features
	integration.EnableFeature("spatial", false)
	integration.EnableFeature("emotional", false)
	integration.EnableFeature("resonance", false)

	ctx := context.Background()
	enhanced, err := integration.PreProcess(ctx, "test")
	if err != nil {
		t.Fatalf("PreProcess failed: %v", err)
	}

	// With all features disabled, enhanced should equal input
	if enhanced != "test" {
		t.Errorf("Expected enhanced='test' with features disabled, got '%s'", enhanced)
	}

	// Re-enable and test
	integration.EnableFeature("spatial", true)
	enhanced, err = integration.PreProcess(ctx, "test")
	if err != nil {
		t.Fatalf("PreProcess failed: %v", err)
	}

	// With spatial enabled, should have markers
	if enhanced == "test" {
		t.Error("Expected enhanced input with spatial markers")
	}
}

// TestCognitiveIntegrationHistory tests history tracking
func TestCognitiveIntegrationHistory(t *testing.T) {
	config := &mockConfig{
		data: map[string]interface{}{
			"embedding_length":                   512,
			"attention.head_count":              8,
			"attention.head_count_kv":           8,
			"block_count":                       6,
			"tokenizer.ggml.model":              "gpt2",
			"tokenizer.ggml.tokens":             []string{"<pad>"},
			"tokenizer.ggml.token_type":         []int{0},
			"tokenizer.ggml.merges":             []string{},
			"tokenizer.ggml.add_bos_token":      true,
			"tokenizer.ggml.bos_token_id":       0,
		},
	}

	model, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create Echo model: %v", err)
	}

	echoModel := model.(*Model)
	identity := deeptreeecho.NewIdentity("TestIdentity")
	integration := NewCognitiveIntegration(echoModel, identity)

	ctx := context.Background()

	// Process multiple inputs
	for i := 0; i < 5; i++ {
		_, err := integration.PreProcess(ctx, "test input")
		if err != nil {
			t.Fatalf("PreProcess failed: %v", err)
		}
	}

	// Check spatial history
	spatialHistory := integration.GetSpatialHistory(10)
	if len(spatialHistory) != 5 {
		t.Errorf("Expected 5 spatial snapshots, got %d", len(spatialHistory))
	}

	// Check emotional history
	emotionalHistory := integration.GetEmotionalHistory(10)
	if len(emotionalHistory) != 5 {
		t.Errorf("Expected 5 emotional snapshots, got %d", len(emotionalHistory))
	}

	// Check resonance history
	resonanceHistory := integration.GetResonanceHistory(10)
	if len(resonanceHistory) != 5 {
		t.Errorf("Expected 5 resonance values, got %d", len(resonanceHistory))
	}

	// Test reset
	integration.Reset()
	metrics := integration.GetMetrics()
	if metrics.InferenceCount != 0 {
		t.Errorf("Expected inference count = 0 after reset, got %d", metrics.InferenceCount)
	}

	spatialHistory = integration.GetSpatialHistory(10)
	if len(spatialHistory) != 0 {
		t.Errorf("Expected 0 spatial snapshots after reset, got %d", len(spatialHistory))
	}
}

// TestCognitiveIntegrationMetrics tests metrics collection
func TestCognitiveIntegrationMetrics(t *testing.T) {
	config := &mockConfig{
		data: map[string]interface{}{
			"embedding_length":                   512,
			"attention.head_count":              8,
			"attention.head_count_kv":           8,
			"block_count":                       6,
			"tokenizer.ggml.model":              "gpt2",
			"tokenizer.ggml.tokens":             []string{"<pad>"},
			"tokenizer.ggml.token_type":         []int{0},
			"tokenizer.ggml.merges":             []string{},
			"tokenizer.ggml.add_bos_token":      true,
			"tokenizer.ggml.bos_token_id":       0,
		},
	}

	model, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create Echo model: %v", err)
	}

	echoModel := model.(*Model)
	identity := deeptreeecho.NewIdentity("TestIdentity")
	integration := NewCognitiveIntegration(echoModel, identity)

	ctx := context.Background()

	// Process some inputs and outputs
	for i := 0; i < 3; i++ {
		_, _ = integration.PreProcess(ctx, "input")
		_, _ = integration.PostProcess(ctx, "output")
	}

	// Get metrics
	metrics := integration.GetMetrics()

	// Verify metrics structure
	if metrics.InferenceCount != 3 {
		t.Errorf("Expected inference count = 3, got %d", metrics.InferenceCount)
	}

	if metrics.SpatialSnapshots != 3 {
		t.Errorf("Expected 3 spatial snapshots, got %d", metrics.SpatialSnapshots)
	}

	if metrics.EmotionalSnapshots != 3 {
		t.Errorf("Expected 3 emotional snapshots, got %d", metrics.EmotionalSnapshots)
	}

	// Coherence should be in valid range
	if metrics.Coherence < 0 || metrics.Coherence > 1 {
		t.Errorf("Expected coherence in [0,1], got %f", metrics.Coherence)
	}

	// Average resonance should be in valid range
	if metrics.AverageResonance < 0 || metrics.AverageResonance > 1 {
		t.Errorf("Expected average resonance in [0,1], got %f", metrics.AverageResonance)
	}

	// Should have an emotion
	if metrics.CurrentEmotion == "" {
		t.Error("Expected non-empty current emotion")
	}
}
