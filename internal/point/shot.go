package point

import (
	"sort"

	"github.com/liamp/tennis-tagger/internal/bridge"
)

// Shot represents a single hit in a rally.
type Shot struct {
	Index      int            // 1-based shot index in rally
	Hitter     int            // 0 = near player (bottom), 1 = far player (top)
	StartFrame int
	EndFrame   int
	Bounce     *bridge.Bounce // where it landed (nil if no bounce detected)
	IsServe    bool
	SpeedKPH   float64
	Confidence float64
}

// trajBounce pairs a bounce with its parent trajectory.
type trajBounce struct {
	bounce bridge.Bounce
	traj   bridge.TrajectoryResult
}

// SegmentShots takes trajectory results and segments them into individual shots.
//
// Primary signal: bounce court-Y position (hitterFromBounce).
// Fallback: trajectory Vy direction when no bounces are detected.
// If a trajectory has bounces, one Shot per bounce.
// If a trajectory has no bounces, one Shot per trajectory using Vy.
func SegmentShots(trajectories []bridge.TrajectoryResult) []Shot {
	// 1. Collect all bounces, plus bounceless trajectories for Vy fallback.
	var bounced []trajBounce
	var bounceless []bridge.TrajectoryResult

	for _, traj := range trajectories {
		if len(traj.Bounces) > 0 {
			for _, b := range traj.Bounces {
				bounced = append(bounced, trajBounce{bounce: b, traj: traj})
			}
		} else {
			bounceless = append(bounceless, traj)
		}
	}

	// 2. Build shots from bounced trajectories.
	var shots []Shot
	if len(bounced) > 0 {
		sort.Slice(bounced, func(i, j int) bool {
			return bounced[i].bounce.FrameIndex < bounced[j].bounce.FrameIndex
		})
		for i, tb := range bounced {
			b := tb.bounce
			hitter := hitterFromBounce(b.CY)
			startFrame := tb.traj.StartFrame
			if i > 0 {
				startFrame = bounced[i-1].bounce.FrameIndex
			}
			bounceVal := b
			shots = append(shots, Shot{
				Hitter:     hitter,
				StartFrame: startFrame,
				EndFrame:   b.FrameIndex,
				Bounce:     &bounceVal,
				SpeedKPH:   tb.traj.SpeedKPH,
				Confidence: tb.traj.Confidence,
			})
		}
	}

	// 3. Build shots from bounceless trajectories using Vy.
	for _, traj := range bounceless {
		hitter := hitterFromVelocity(traj.Vy)
		shots = append(shots, Shot{
			Hitter:     hitter,
			StartFrame: traj.StartFrame,
			EndFrame:   traj.EndFrame,
			SpeedKPH:   traj.SpeedKPH,
			Confidence: traj.Confidence,
		})
	}

	if len(shots) == 0 {
		return nil
	}

	// 4. Sort all shots chronologically.
	sort.Slice(shots, func(i, j int) bool {
		return shots[i].StartFrame < shots[j].StartFrame
	})

	// 5. Set indices and serve flag.
	for i := range shots {
		shots[i].Index = i + 1
		shots[i].IsServe = (i == 0)
	}

	return shots
}

// hitterFromBounce returns the player who hit the ball based on bounce position.
func hitterFromBounce(cy float64) int {
	if cy < NetY {
		return 1
	}
	return 0
}

// hitterFromVelocity returns the player who hit the ball based on trajectory direction.
// vy > 0 (moving toward far baseline) → near player (0) hit it
// vy <= 0 (moving toward near baseline) → far player (1) hit it
func hitterFromVelocity(vy float64) int {
	if vy > 0 {
		return 0
	}
	return 1
}
