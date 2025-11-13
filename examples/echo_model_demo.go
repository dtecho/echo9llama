package main

import (
	"context"
	"fmt"
	"log"

	"github.com/EchoCog/echollama/core/deeptreeecho"
	"github.com/EchoCog/echollama/model/models/echo"
)

// This example demonstrates how to use the Echo model with Deep Tree Echo integration
func main() {
	fmt.Println("🌊 Echo Model Integration Example")
	fmt.Println("==================================\n")

	// Step 1: Create a minimal Echo model configuration
	config := &mockConfig{
		data: map[string]interface{}{
			"embedding_length":                   768,
			"attention.head_count":              12,
			"attention.head_count_kv":           12,
			"block_count":                       12,
			"attention.layer_norm_rms_epsilon":  1e-6,
			"tokenizer.ggml.model":              "gpt2",
			"tokenizer.ggml.tokens":             []string{"<pad>", "hello", "world", "!"},
			"tokenizer.ggml.token_type":         []int{0, 1, 1, 1},
			"tokenizer.ggml.merges":             []string{},
			"tokenizer.ggml.add_bos_token":      true,
			"tokenizer.ggml.bos_token_id":       0,
			"tokenizer.ggml.add_eos_token":      false,
			"tokenizer.ggml.eos_token_id":       0,
			// Echo-specific configuration
			"echo.reservoir_size":               1536,
			"echo.spatial_dim":                  3,
			"echo.emotional_dim":                8,
			"echo.memory_capacity":              1024,
			"echo.spectral_radius":              0.95,
			"echo.resonance_threshold":          0.7,
		},
	}

	fmt.Println("Step 1: Creating Echo model...")
	model, err := echo.New(config)
	if err != nil {
		log.Fatalf("Failed to create Echo model: %v", err)
	}

	echoModel := model.(*echo.Model)
	fmt.Printf("✓ Created Echo model: %s\n\n", echoModel.String())

	// Step 2: Create Deep Tree Echo identity
	fmt.Println("Step 2: Initializing Deep Tree Echo identity...")
	identity := deeptreeecho.NewIdentity("EchoExample")
	fmt.Printf("✓ Identity initialized: %s\n", identity.Name)
	fmt.Printf("  Coherence: %.2f%%\n", identity.Coherence*100)
	fmt.Printf("  Emotion: %s\n\n", identity.EmotionalState.Primary.Type)

	// Step 3: Create cognitive integration layer
	fmt.Println("Step 3: Creating cognitive integration...")
	integration := echo.NewCognitiveIntegration(echoModel, identity)
	fmt.Println("✓ Cognitive integration created\n")

	// Step 4: Demonstrate pre-processing with cognitive context
	fmt.Println("Step 4: Processing input with cognitive enhancement...")
	ctx := context.Background()
	input := "Hello, world! How does Deep Tree Echo enhance AI?"

	enhanced, err := integration.PreProcess(ctx, input)
	if err != nil {
		log.Fatalf("Pre-processing failed: %v", err)
	}

	fmt.Printf("Original input:  %s\n", input)
	fmt.Printf("Enhanced input:  %s\n\n", enhanced)

	// Step 5: Simulate model inference (in real use, this would run the actual model)
	fmt.Println("Step 5: Simulating model inference...")
	simulatedOutput := "Deep Tree Echo enhances AI through embodied cognition, " +
		"integrating spatial awareness, emotional dynamics, and reservoir computing " +
		"for more human-like cognitive processing."
	fmt.Printf("Model output: %s\n\n", simulatedOutput)

	// Step 6: Post-process output with cognitive insights
	fmt.Println("Step 6: Post-processing with cognitive insights...")
	processed, err := integration.PostProcess(ctx, simulatedOutput)
	if err != nil {
		log.Fatalf("Post-processing failed: %v", err)
	}

	fmt.Printf("Processed output: %s\n\n", processed)

	// Step 7: Get cognitive metrics
	fmt.Println("Step 7: Retrieving cognitive metrics...")
	metrics := integration.GetMetrics()
	fmt.Printf("Inference Count:     %d\n", metrics.InferenceCount)
	fmt.Printf("Average Resonance:   %.3f\n", metrics.AverageResonance)
	fmt.Printf("Coherence:           %.2f%%\n", metrics.Coherence*100)
	fmt.Printf("Current Emotion:     %s\n", metrics.CurrentEmotion)
	fmt.Printf("Spatial Snapshots:   %d\n", metrics.SpatialSnapshots)
	fmt.Printf("Emotional Snapshots: %d\n", metrics.EmotionalSnapshots)
	fmt.Printf("Memory Nodes:        %d\n", metrics.MemoryNodes)
	fmt.Printf("Reservoir Echo:      %.3f\n\n", metrics.ReservoirEcho)

	// Step 8: Access history
	fmt.Println("Step 8: Accessing cognitive history...")
	spatialHistory := integration.GetSpatialHistory(10)
	fmt.Printf("Spatial history entries: %d\n", len(spatialHistory))
	if len(spatialHistory) > 0 {
		latest := spatialHistory[len(spatialHistory)-1]
		fmt.Printf("Latest position: (%.2f, %.2f, %.2f)\n",
			latest.Position[0], latest.Position[1], latest.Position[2])
		fmt.Printf("Field resonance: %.3f\n", latest.Field.Resonance)
	}

	emotionalHistory := integration.GetEmotionalHistory(10)
	fmt.Printf("\nEmotional history entries: %d\n", len(emotionalHistory))
	if len(emotionalHistory) > 0 {
		latest := emotionalHistory[len(emotionalHistory)-1]
		fmt.Printf("Latest emotion: %s (intensity: %.2f)\n",
			latest.Primary, latest.Intensity)
	}

	resonanceHistory := integration.GetResonanceHistory(10)
	fmt.Printf("\nResonance history entries: %d\n", len(resonanceHistory))
	if len(resonanceHistory) > 0 {
		fmt.Printf("Recent resonance values: %v\n", resonanceHistory)
	}

	// Step 9: Feature control demonstration
	fmt.Println("\nStep 9: Demonstrating feature control...")
	fmt.Println("Disabling emotional processing...")
	integration.EnableFeature("emotional", false)

	input2 := "Testing with emotion disabled"
	enhanced2, _ := integration.PreProcess(ctx, input2)
	fmt.Printf("Input with emotion disabled: %s\n", enhanced2)

	fmt.Println("\nRe-enabling all features...")
	integration.EnableFeature("spatial", true)
	integration.EnableFeature("emotional", true)
	integration.EnableFeature("resonance", true)
	integration.EnableFeature("memory", true)

	// Step 10: Reset and verify
	fmt.Println("\nStep 10: Resetting integration state...")
	integration.Reset()
	metrics = integration.GetMetrics()
	fmt.Printf("After reset - Inference count: %d\n", metrics.InferenceCount)
	fmt.Printf("After reset - Spatial snapshots: %d\n", metrics.SpatialSnapshots)

	fmt.Println("\n🌊 Example complete! The Echo model successfully integrates")
	fmt.Println("   Deep Tree Echo's cognitive architecture for enhanced AI processing.")
	fmt.Println("\n✨ Key Benefits:")
	fmt.Println("   • Spatial awareness in 3D cognitive space")
	fmt.Println("   • Emotional dynamics and valence tracking")
	fmt.Println("   • Reservoir computing for temporal patterns")
	fmt.Println("   • Memory consolidation and learning")
	fmt.Println("   • Resonance-based pattern activation")
}

// mockConfig implements fs.Config for the example
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
