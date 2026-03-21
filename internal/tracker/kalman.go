package tracker

import (
	"gonum.org/v1/gonum/mat"
)

// KalmanFilter implements a 7D constant-velocity Kalman filter for bounding box tracking.
// State vector: [x, y, w, h, vx, vy, vw] where (x,y) is center, (w,h) is size.
// Measurement vector: [x, y, w, h].
type KalmanFilter struct {
	x *mat.VecDense // state vector (7x1)
	P *mat.Dense    // state covariance (7x7)
	F *mat.Dense    // transition matrix (7x7)
	H *mat.Dense    // measurement matrix (4x7)
	R *mat.Dense    // measurement noise (4x4)
	Q *mat.Dense    // process noise (7x7)
}

// NewKalmanFilter creates a new Kalman filter initialized from a bounding box [x, y, w, h].
func NewKalmanFilter(bbox [4]float64) *KalmanFilter {
	kf := &KalmanFilter{}

	// State vector: [x, y, w, h, vx, vy, vw], velocities start at 0
	kf.x = mat.NewVecDense(7, []float64{
		bbox[0], bbox[1], bbox[2], bbox[3],
		0, 0, 0,
	})

	// Transition matrix F: identity with velocity coupling
	// x' = x + vx, y' = y + vy, w' = w + vw
	fData := make([]float64, 49)
	for i := 0; i < 7; i++ {
		fData[i*7+i] = 1 // identity diagonal
	}
	fData[0*7+4] = 1 // F[0,4] = 1: x += vx
	fData[1*7+5] = 1 // F[1,5] = 1: y += vy
	fData[2*7+6] = 1 // F[2,6] = 1: w += vw
	kf.F = mat.NewDense(7, 7, fData)

	// Measurement matrix H: maps state to measurement [x, y, w, h]
	hData := make([]float64, 28)
	hData[0*7+0] = 1 // H[0,0] = 1
	hData[1*7+1] = 1 // H[1,1] = 1
	hData[2*7+2] = 1 // H[2,2] = 1
	hData[3*7+3] = 1 // H[3,3] = 1
	kf.H = mat.NewDense(4, 7, hData)

	// Measurement noise R: 10 * I(4)
	rData := make([]float64, 16)
	for i := 0; i < 4; i++ {
		rData[i*4+i] = 10
	}
	kf.R = mat.NewDense(4, 4, rData)

	// Process noise Q: I(7) with velocity components scaled by 0.01
	qData := make([]float64, 49)
	for i := 0; i < 4; i++ {
		qData[i*7+i] = 1 // position components
	}
	for i := 4; i < 7; i++ {
		qData[i*7+i] = 0.01 // velocity components
	}
	kf.Q = mat.NewDense(7, 7, qData)

	// Initial covariance P: 10 * I(7), velocity diagonal *= 1000
	pData := make([]float64, 49)
	for i := 0; i < 4; i++ {
		pData[i*7+i] = 10 // position uncertainty
	}
	for i := 4; i < 7; i++ {
		pData[i*7+i] = 10 * 1000 // velocity uncertainty = 10000
	}
	kf.P = mat.NewDense(7, 7, pData)

	return kf
}

// Predict advances the state using the constant velocity model.
func (kf *KalmanFilter) Predict() {
	// x = F * x
	var newX mat.VecDense
	newX.MulVec(kf.F, kf.x)
	kf.x.CopyVec(&newX)

	// P = F * P * F^T + Q
	var fp, fpft mat.Dense
	fp.Mul(kf.F, kf.P)
	fpft.Mul(&fp, kf.F.T())
	fpft.Add(&fpft, kf.Q)
	kf.P.Copy(&fpft)
}

// Update performs the Kalman update step with a measurement [x, y, w, h].
func (kf *KalmanFilter) Update(measurement [4]float64) {
	z := mat.NewVecDense(4, measurement[:])

	// y = z - H * x (innovation)
	var hx mat.VecDense
	hx.MulVec(kf.H, kf.x)
	var y mat.VecDense
	y.SubVec(z, &hx)

	// S = H * P * H^T + R (innovation covariance)
	var hp, s mat.Dense
	hp.Mul(kf.H, kf.P)
	s.Mul(&hp, kf.H.T())
	s.Add(&s, kf.R)

	// K = P * H^T * S^{-1} (Kalman gain)
	var sInv, pht, k mat.Dense
	err := sInv.Inverse(&s)
	if err != nil {
		// If S is singular, skip update
		return
	}
	pht.Mul(kf.P, kf.H.T())
	k.Mul(&pht, &sInv)

	// x = x + K * y
	var ky mat.VecDense
	ky.MulVec(&k, &y)
	kf.x.AddVec(kf.x, &ky)

	// P = (I - K * H) * P
	var kh, iMinusKH mat.Dense
	kh.Mul(&k, kf.H)
	eye := mat.NewDense(7, 7, nil)
	for i := 0; i < 7; i++ {
		eye.Set(i, i, 1)
	}
	iMinusKH.Sub(eye, &kh)
	var newP mat.Dense
	newP.Mul(&iMinusKH, kf.P)
	kf.P.Copy(&newP)
}

// State returns the current 7D state vector [x, y, w, h, vx, vy, vw].
func (kf *KalmanFilter) State() [7]float64 {
	var s [7]float64
	for i := 0; i < 7; i++ {
		s[i] = kf.x.AtVec(i)
	}
	return s
}

// BBox returns the current bounding box [x, y, w, h] from the state.
func (kf *KalmanFilter) BBox() [4]float64 {
	return [4]float64{
		kf.x.AtVec(0),
		kf.x.AtVec(1),
		kf.x.AtVec(2),
		kf.x.AtVec(3),
	}
}
