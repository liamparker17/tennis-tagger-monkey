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

// trajBounce pairs a bounce with its parent trajectory so speed/confidence can
// be propagated to the resulting Shot.
type trajBounce struct {
	bounce bridge.Bounce
	traj   bridge.TrajectoryResult
}

// SegmentShots takes trajectory results and segments them into individual shots.
//
// Each detected bounce defines one shot. The hitter of a shot is determined by
// which half of the court the ball landed in:
//   - bounce.CY < NetY  → ball was heading toward the near baseline → far player (1) hit it
//   - bounce.CY >= NetY → ball was heading toward the far baseline  → near player (0) hit it
//
// The first shot in the sequence is always marked as a serve (IsServe = true).
func SegmentShots(trajectories []bridge.TrajectoryResult) []Shot {
	// 1. Collect all bounces across all trajectories.
	var all []trajBounce
	for _, traj := range trajectories {
		for _, b := range traj.Bounces {
			all = append(all, trajBounce{bounce: b, traj: traj})
		}
	}

	if len(all) == 0 {
		return nil
	}

	// 2. Sort by frame index so shots are in chronological order.
	sort.Slice(all, func(i, j int) bool {
		return all[i].bounce.FrameIndex < all[j].bounce.FrameIndex
	})

	// 3. Build one Shot per bounce.
	shots := make([]Shot, len(all))
	for i, tb := range all {
		b := tb.bounce // local copy so we can safely take its address below

		hitter := hitterFromBounce(b.CY)

		// Frame span: from previous bounce's frame (or traj start) to this bounce.
		startFrame := tb.traj.StartFrame
		if i > 0 {
			startFrame = all[i-1].bounce.FrameIndex
		}

		// Store a pointer into the shots slice's own copy of the bounce so the
		// caller gets a stable pointer that does not alias the loop variable.
		shots[i] = Shot{
			Index:      i + 1,
			Hitter:     hitter,
			StartFrame: startFrame,
			EndFrame:   b.FrameIndex,
			IsServe:    i == 0,
			SpeedKPH:   tb.traj.SpeedKPH,
			Confidence: tb.traj.Confidence,
		}
		// Point Bounce at the copy stored inside the slice element.
		bounceVal := b
		shots[i].Bounce = &bounceVal
	}

	return shots
}

// hitterFromBounce returns the player who hit the ball that produced a bounce
// at the given court-Y coordinate.
//
//   - cy < NetY  → ball landed on near side → far player (1) sent it there
//   - cy >= NetY → ball landed on far side  → near player (0) sent it there
func hitterFromBounce(cy float64) int {
	if cy < NetY {
		return 1
	}
	return 0
}
