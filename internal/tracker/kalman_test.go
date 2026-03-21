package tracker

import (
	"math"
	"testing"
)

func TestKalmanFilter_Predict(t *testing.T) {
	// With zero velocity, position should remain stable after predict.
	bbox := [4]float64{100, 200, 50, 80}
	kf := NewKalmanFilter(bbox)

	stateBefore := kf.State()
	kf.Predict()
	stateAfter := kf.State()

	// Position components should be unchanged (velocity is 0)
	for i := 0; i < 4; i++ {
		if math.Abs(stateAfter[i]-stateBefore[i]) > 1e-9 {
			t.Errorf("Position component %d changed: before=%f, after=%f", i, stateBefore[i], stateAfter[i])
		}
	}
}

func TestKalmanFilter_Update(t *testing.T) {
	// After update with a different measurement, state should move toward it.
	bbox := [4]float64{100, 200, 50, 80}
	kf := NewKalmanFilter(bbox)

	measurement := [4]float64{120, 210, 55, 85}
	kf.Predict()
	kf.Update(measurement)

	state := kf.State()

	// x should move toward 120 (from initial 100)
	if state[0] <= 100 || state[0] >= 120 {
		t.Errorf("Expected x between 100 and 120, got %f", state[0])
	}
	// y should move toward 210 (from initial 200)
	if state[1] <= 200 || state[1] >= 210 {
		t.Errorf("Expected y between 200 and 210, got %f", state[1])
	}
	// w should move toward 55 (from initial 50)
	if state[2] <= 50 || state[2] >= 55 {
		t.Errorf("Expected w between 50 and 55, got %f", state[2])
	}
	// h should move toward 85 (from initial 80)
	if state[3] <= 80 || state[3] >= 85 {
		t.Errorf("Expected h between 80 and 85, got %f", state[3])
	}
}

func TestKalmanFilter_VelocityEstimation(t *testing.T) {
	// Object moving right at 5 px/frame for 10 frames.
	// Kalman filter should estimate vx close to 5.
	startX := 100.0
	bbox := [4]float64{startX, 200, 50, 80}
	kf := NewKalmanFilter(bbox)

	for i := 1; i <= 10; i++ {
		kf.Predict()
		measurement := [4]float64{startX + float64(i)*5, 200, 50, 80}
		kf.Update(measurement)
	}

	state := kf.State()
	vx := state[4]

	// vx should be approximately 5
	if math.Abs(vx-5) > 1.0 {
		t.Errorf("Expected vx ~ 5, got %f", vx)
	}

	// vy should be approximately 0
	vy := state[5]
	if math.Abs(vy) > 1.0 {
		t.Errorf("Expected vy ~ 0, got %f", vy)
	}
}
