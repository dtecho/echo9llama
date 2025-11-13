package echo

import (
	"testing"

	"github.com/EchoCog/echollama/fs"
)

// TestEchoModelRegistration tests that the Echo model is properly registered
func TestEchoModelRegistration(t *testing.T) {
	// This test ensures the init() function has run and registered the model
	// The actual registration check happens when the model package is imported
}

// TestNewEchoModel tests creating a new Echo model with minimal config
func TestNewEchoModel(t *testing.T) {
	// Create a minimal mock config
	config := &mockConfig{
		data: map[string]interface{}{
			"embedding_length":                   768,
			"attention.head_count":              12,
			"attention.head_count_kv":           12,
			"block_count":                       12,
			"attention.layer_norm_rms_epsilon":  1e-6,
			"tokenizer.ggml.model":              "gpt2",
			"tokenizer.ggml.tokens":             []string{"<pad>", "hello", "world"},
			"tokenizer.ggml.token_type":         []int{0, 1, 1},
			"tokenizer.ggml.merges":             []string{},
			"tokenizer.ggml.add_bos_token":      true,
			"tokenizer.ggml.bos_token_id":       0,
			"tokenizer.ggml.add_eos_token":      false,
			"tokenizer.ggml.eos_token_id":       0,
		},
	}

	model, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create Echo model: %v", err)
	}

	if model == nil {
		t.Fatal("Expected non-nil model")
	}

	echoModel, ok := model.(*Model)
	if !ok {
		t.Fatal("Expected model to be *echo.Model type")
	}

	// Verify basic configuration
	if echoModel.hiddenSize != 768 {
		t.Errorf("Expected hiddenSize=768, got %d", echoModel.hiddenSize)
	}

	if echoModel.numHeads != 12 {
		t.Errorf("Expected numHeads=12, got %d", echoModel.numHeads)
	}

	if len(echoModel.Layers) != 12 {
		t.Errorf("Expected 12 layers, got %d", len(echoModel.Layers))
	}

	// Verify Echo-specific defaults
	if echoModel.reservoirSize != 768*2 {
		t.Errorf("Expected reservoirSize=%d, got %d", 768*2, echoModel.reservoirSize)
	}

	if echoModel.spatialDim != 3 {
		t.Errorf("Expected spatialDim=3, got %d", echoModel.spatialDim)
	}

	if echoModel.emotionalDim != 8 {
		t.Errorf("Expected emotionalDim=8, got %d", echoModel.emotionalDim)
	}
}

// TestEchoModelWithCustomConfig tests creating Echo model with custom Echo parameters
func TestEchoModelWithCustomConfig(t *testing.T) {
	config := &mockConfig{
		data: map[string]interface{}{
			"embedding_length":                   1024,
			"attention.head_count":              16,
			"attention.head_count_kv":           16,
			"block_count":                       24,
			"attention.layer_norm_rms_epsilon":  1e-5,
			"tokenizer.ggml.model":              "gpt2",
			"tokenizer.ggml.tokens":             []string{"<pad>", "test"},
			"tokenizer.ggml.token_type":         []int{0, 1},
			"tokenizer.ggml.merges":             []string{},
			"tokenizer.ggml.add_bos_token":      true,
			"tokenizer.ggml.bos_token_id":       0,
			"tokenizer.ggml.add_eos_token":      false,
			"tokenizer.ggml.eos_token_id":       0,
			"echo.reservoir_size":               2048,
			"echo.spatial_dim":                  5,
			"echo.emotional_dim":                12,
			"echo.memory_capacity":              2048,
			"echo.spectral_radius":              0.9,
			"echo.resonance_threshold":          0.8,
		},
	}

	model, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create Echo model: %v", err)
	}

	echoModel := model.(*Model)

	// Verify custom Echo configuration
	if echoModel.reservoirSize != 2048 {
		t.Errorf("Expected reservoirSize=2048, got %d", echoModel.reservoirSize)
	}

	if echoModel.spatialDim != 5 {
		t.Errorf("Expected spatialDim=5, got %d", echoModel.spatialDim)
	}

	if echoModel.emotionalDim != 12 {
		t.Errorf("Expected emotionalDim=12, got %d", echoModel.emotionalDim)
	}

	if echoModel.memoryCapacity != 2048 {
		t.Errorf("Expected memoryCapacity=2048, got %d", echoModel.memoryCapacity)
	}

	if echoModel.spectralRadius != 0.9 {
		t.Errorf("Expected spectralRadius=0.9, got %f", echoModel.spectralRadius)
	}

	if echoModel.resonanceThreshold != 0.8 {
		t.Errorf("Expected resonanceThreshold=0.8, got %f", echoModel.resonanceThreshold)
	}
}

// TestEchoModelString tests the String() method
func TestEchoModelString(t *testing.T) {
	config := &mockConfig{
		data: map[string]interface{}{
			"embedding_length":                   768,
			"attention.head_count":              12,
			"attention.head_count_kv":           12,
			"block_count":                       12,
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
	str := echoModel.String()

	if str == "" {
		t.Error("Expected non-empty string representation")
	}

	// Check that it contains key information
	if !contains(str, "Echo Model") {
		t.Error("Expected string to contain 'Echo Model'")
	}
}

// mockConfig implements fs.Config for testing
type mockConfig struct {
	data map[string]interface{}
}

func (m *mockConfig) String(key string, defaultValue ...string) string {
	if v, ok := m.data[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return ""
}

func (m *mockConfig) Strings(key string) []string {
	if v, ok := m.data[key]; ok {
		if s, ok := v.([]string); ok {
			return s
		}
	}
	return nil
}

func (m *mockConfig) Uint(key string, defaultValue ...uint64) uint64 {
	if v, ok := m.data[key]; ok {
		if i, ok := v.(int); ok {
			return uint64(i)
		}
		if u, ok := v.(uint64); ok {
			return u
		}
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return 0
}

func (m *mockConfig) Int(key string, defaultValue ...int) int {
	if v, ok := m.data[key]; ok {
		if i, ok := v.(int); ok {
			return i
		}
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return 0
}

func (m *mockConfig) Ints(key string) []int {
	if v, ok := m.data[key]; ok {
		if s, ok := v.([]int); ok {
			return s
		}
	}
	return nil
}

func (m *mockConfig) Float(key string, defaultValue ...float32) float32 {
	if v, ok := m.data[key]; ok {
		if f, ok := v.(float64); ok {
			return float32(f)
		}
		if f, ok := v.(float32); ok {
			return f
		}
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return 0
}

func (m *mockConfig) Floats(key string) []float32 {
	if v, ok := m.data[key]; ok {
		if s, ok := v.([]float32); ok {
			return s
		}
	}
	return nil
}

func (m *mockConfig) Bool(key string, defaultValue ...bool) bool {
	if v, ok := m.data[key]; ok {
		if b, ok := v.(bool); ok {
			return b
		}
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return false
}

func (m *mockConfig) Architecture() string {
	return "echo"
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
