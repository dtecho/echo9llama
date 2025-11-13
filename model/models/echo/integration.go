package echo

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/EchoCog/echollama/core/deeptreeecho"
)

// CognitiveIntegration provides integration between Echo model and Deep Tree Echo
type CognitiveIntegration struct {
	mu       sync.RWMutex
	identity *deeptreeecho.Identity
	model    *Model
	
	// Cognitive state tracking
	spatialHistory    []SpatialSnapshot
	emotionalHistory  []EmotionalSnapshot
	resonanceHistory  []float64
	
	// Integration settings
	enableSpatial     bool
	enableEmotional   bool
	enableResonance   bool
	enableMemory      bool
	
	// Performance metrics
	inferenceCount    uint64
	totalLatency      time.Duration
	avgResonance      float64
}

// SpatialSnapshot captures spatial context at a point in time
type SpatialSnapshot struct {
	Timestamp  time.Time
	Position   [3]float64 // X, Y, Z coordinates
	Orientation [4]float64 // Quaternion W, X, Y, Z
	Field      FieldState
}

// FieldState captures the cognitive field state
type FieldState struct {
	Intensity  float64
	Gradient   [3]float64
	Curvature  float64
	Resonance  float64
}

// EmotionalSnapshot captures emotional state at a point in time
type EmotionalSnapshot struct {
	Timestamp   time.Time
	Primary     string
	Intensity   float64
	Valence     float64
	Arousal     float64
}

// NewCognitiveIntegration creates a new cognitive integration layer
func NewCognitiveIntegration(model *Model, identity *deeptreeecho.Identity) *CognitiveIntegration {
	return &CognitiveIntegration{
		identity:          identity,
		model:             model,
		spatialHistory:    make([]SpatialSnapshot, 0, 1000),
		emotionalHistory:  make([]EmotionalSnapshot, 0, 1000),
		resonanceHistory:  make([]float64, 0, 1000),
		enableSpatial:     true,
		enableEmotional:   true,
		enableResonance:   true,
		enableMemory:      true,
	}
}

// PreProcess prepares input for the Echo model with cognitive context
func (ci *CognitiveIntegration) PreProcess(ctx context.Context, input string) (string, error) {
	ci.mu.Lock()
	defer ci.mu.Unlock()
	
	if ci.identity == nil {
		return input, nil // No cognitive enhancement if no identity
	}
	
	// Process input through Deep Tree Echo
	ci.identity.Process(input)
	
	// Add cognitive context markers
	enhanced := input
	
	if ci.enableSpatial {
		spatial := ci.captureSpatialSnapshot()
		ci.spatialHistory = append(ci.spatialHistory, spatial)
		enhanced = fmt.Sprintf("[SPATIAL:%.2f,%.2f,%.2f] %s", 
			spatial.Position[0], spatial.Position[1], spatial.Position[2], enhanced)
	}
	
	if ci.enableEmotional {
		emotional := ci.captureEmotionalSnapshot()
		ci.emotionalHistory = append(ci.emotionalHistory, emotional)
		enhanced = fmt.Sprintf("[EMOTION:%s:%.2f] %s",
			emotional.Primary, emotional.Intensity, enhanced)
	}
	
	if ci.enableResonance {
		resonance := ci.identity.SpatialContext.Field.Resonance
		ci.resonanceHistory = append(ci.resonanceHistory, resonance)
		enhanced = fmt.Sprintf("[RESONANCE:%.3f] %s", resonance, enhanced)
	}
	
	return enhanced, nil
}

// PostProcess enhances output with cognitive insights
func (ci *CognitiveIntegration) PostProcess(ctx context.Context, output string) (string, error) {
	ci.mu.Lock()
	defer ci.mu.Unlock()
	
	if ci.identity == nil {
		return output, nil
	}
	
	// Process output through Deep Tree Echo
	ci.identity.Process(output)
	
	// Add cognitive insights if enabled
	enhanced := output
	
	// Calculate current resonance
	resonance := ci.calculateResonance()
	ci.avgResonance = (ci.avgResonance*float64(ci.inferenceCount) + resonance) / float64(ci.inferenceCount+1)
	ci.inferenceCount++
	
	// Add resonance indicator for high-resonance responses
	if resonance > ci.model.resonanceThreshold {
		enhanced = fmt.Sprintf("✨ %s", enhanced)
	}
	
	// Store in memory if significant
	if ci.enableMemory && (resonance > ci.model.resonanceThreshold || len(output) > 100) {
		ci.identity.Remember(fmt.Sprintf("response_%d", ci.inferenceCount), output)
	}
	
	return enhanced, nil
}

// captureSpatialSnapshot captures current spatial state
func (ci *CognitiveIntegration) captureSpatialSnapshot() SpatialSnapshot {
	spatial := ci.identity.SpatialContext
	return SpatialSnapshot{
		Timestamp: time.Now(),
		Position: [3]float64{
			spatial.Position.X,
			spatial.Position.Y,
			spatial.Position.Z,
		},
		Orientation: [4]float64{
			spatial.Orientation.W,
			spatial.Orientation.X,
			spatial.Orientation.Y,
			spatial.Orientation.Z,
		},
		Field: FieldState{
			Intensity: spatial.Field.Intensity,
			Gradient: [3]float64{
				spatial.Field.Gradient.X,
				spatial.Field.Gradient.Y,
				spatial.Field.Gradient.Z,
			},
			Curvature: spatial.Field.Curvature,
			Resonance: spatial.Field.Resonance,
		},
	}
}

// captureEmotionalSnapshot captures current emotional state
func (ci *CognitiveIntegration) captureEmotionalSnapshot() EmotionalSnapshot {
	emotional := ci.identity.EmotionalState
	return EmotionalSnapshot{
		Timestamp: time.Now(),
		Primary:   emotional.Primary.Type,
		Intensity: emotional.Intensity,
		Valence:   emotional.Valence,
		Arousal:   emotional.Arousal,
	}
}

// calculateResonance calculates current resonance level
func (ci *CognitiveIntegration) calculateResonance() float64 {
	if ci.identity == nil || ci.identity.SpatialContext == nil {
		return 0.0
	}
	
	// Combine multiple resonance factors
	spatialResonance := ci.identity.SpatialContext.Field.Resonance
	coherence := ci.identity.Coherence
	
	// Weighted average
	return (spatialResonance*0.7 + coherence*0.3)
}

// GetMetrics returns performance and cognitive metrics
func (ci *CognitiveIntegration) GetMetrics() CognitiveMetrics {
	ci.mu.RLock()
	defer ci.mu.RUnlock()
	
	return CognitiveMetrics{
		InferenceCount:    ci.inferenceCount,
		AverageResonance:  ci.avgResonance,
		SpatialSnapshots:  len(ci.spatialHistory),
		EmotionalSnapshots: len(ci.emotionalHistory),
		MemoryNodes:       len(ci.identity.Memory.Nodes),
		Coherence:         ci.identity.Coherence,
		CurrentEmotion:    ci.identity.EmotionalState.Primary.Type,
		ReservoirEcho:     ci.identity.Reservoir.calculateEcho(),
	}
}

// CognitiveMetrics contains metrics about the cognitive integration
type CognitiveMetrics struct {
	InferenceCount     uint64
	AverageResonance   float64
	SpatialSnapshots   int
	EmotionalSnapshots int
	MemoryNodes        int
	Coherence          float64
	CurrentEmotion     string
	ReservoirEcho      float64
}

// Reset resets the integration state
func (ci *CognitiveIntegration) Reset() {
	ci.mu.Lock()
	defer ci.mu.Unlock()
	
	ci.spatialHistory = ci.spatialHistory[:0]
	ci.emotionalHistory = ci.emotionalHistory[:0]
	ci.resonanceHistory = ci.resonanceHistory[:0]
	ci.inferenceCount = 0
	ci.totalLatency = 0
	ci.avgResonance = 0.0
}

// EnableFeature enables or disables a cognitive feature
func (ci *CognitiveIntegration) EnableFeature(feature string, enabled bool) {
	ci.mu.Lock()
	defer ci.mu.Unlock()
	
	switch feature {
	case "spatial":
		ci.enableSpatial = enabled
	case "emotional":
		ci.enableEmotional = enabled
	case "resonance":
		ci.enableResonance = enabled
	case "memory":
		ci.enableMemory = enabled
	}
}

// GetSpatialHistory returns recent spatial snapshots
func (ci *CognitiveIntegration) GetSpatialHistory(limit int) []SpatialSnapshot {
	ci.mu.RLock()
	defer ci.mu.RUnlock()
	
	start := 0
	if len(ci.spatialHistory) > limit {
		start = len(ci.spatialHistory) - limit
	}
	
	history := make([]SpatialSnapshot, len(ci.spatialHistory)-start)
	copy(history, ci.spatialHistory[start:])
	return history
}

// GetEmotionalHistory returns recent emotional snapshots
func (ci *CognitiveIntegration) GetEmotionalHistory(limit int) []EmotionalSnapshot {
	ci.mu.RLock()
	defer ci.mu.RUnlock()
	
	start := 0
	if len(ci.emotionalHistory) > limit {
		start = len(ci.emotionalHistory) - limit
	}
	
	history := make([]EmotionalSnapshot, len(ci.emotionalHistory)-start)
	copy(history, ci.emotionalHistory[start:])
	return history
}

// GetResonanceHistory returns recent resonance values
func (ci *CognitiveIntegration) GetResonanceHistory(limit int) []float64 {
	ci.mu.RLock()
	defer ci.mu.RUnlock()
	
	start := 0
	if len(ci.resonanceHistory) > limit {
		start = len(ci.resonanceHistory) - limit
	}
	
	history := make([]float64, len(ci.resonanceHistory)-start)
	copy(history, ci.resonanceHistory[start:])
	return history
}
