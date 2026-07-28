package sim

import "math/bits"

// Rand is the simulation's only source of randomness.
//
// # Why not math/rand
//
// Three reasons, each of which has broken somebody's reproducibility:
//
// First, math/rand's top-level functions share a global source. Any library in
// the process can draw from it, so the sequence your test sees depends on what
// else happened to run — including test ordering, which changes when you add a
// test.
//
// Second, Go's global source is seeded randomly at startup since Go 1.20, so
// "the same test" is a different test every run unless it is explicitly seeded.
//
// Third and most subtly, the *algorithm* behind math/rand's outputs is not
// guaranteed stable across Go releases. A seed that reproduced a bug on Go 1.24
// may explore a different schedule on 1.25 — and a reproduction recipe that
// expires with a toolchain upgrade is not a reproduction recipe.
//
// So Rand implements its own generator whose algorithm is pinned here, in this
// file, and cannot drift. The choice is splitmix64: a well-studied, extremely
// fast, statistically sound generator that carries all its state in a single
// uint64. That last property is what makes it right for this job — one word of
// state means a schedule can be checkpointed, logged and restored trivially.
//
// splitmix64 is not cryptographically secure and must never be used for
// anything that needs to be unpredictable. It is used here precisely because it
// is predictable.
//
// # The rule
//
// Every random choice in a simulation must come from this type: which key to
// write, how long to delay a message, whether to drop a packet, which node to
// crash. A single call to math/rand or crypto/rand anywhere in the loop produces
// a simulation that finds real bugs and then cannot show them to you again —
// which is worse than one that finds nothing, because it burns trust.
type Rand struct {
	state uint64
}

// NewRand returns a generator seeded with the given value.
func NewRand(seed uint64) *Rand {
	// The golden-ratio increment is splitmix64's standard constant. A zero seed
	// is a legitimate input for this algorithm — unlike some generators, it does
	// not degenerate — so no special case is needed.
	return &Rand{state: seed}
}

// Uint64 returns the next value in the sequence.
func (r *Rand) Uint64() uint64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// Uint32 returns the high 32 bits of the next value.
//
// The HIGH bits, deliberately. Taking the low bits of many generators gives a
// weaker sequence; splitmix64's final xor-shift means both halves are sound, but
// taking the high half is the habit that stays correct if the generator is ever
// swapped.
func (r *Rand) Uint32() uint32 { return uint32(r.Uint64() >> 32) }

// Intn returns a uniformly distributed value in [0, n). It panics if n <= 0.
//
// The implementation is Lemire's multiply-shift with rejection, which avoids the
// modulo bias of `Uint64() % n`. The bias is small for small n but it is
// systematic: with plain modulo, a scheduler choosing among 3 runnable actors
// would favour actor 0 very slightly, forever, and a rare interleaving involving
// actor 2 would be explored measurably less often. A simulator's entire value is
// its coverage of unlikely orderings, so a subtle skew against them is the one
// bias worth spending a few instructions to remove.
func (r *Rand) Intn(n int) int {
	if n <= 0 {
		panic("lockstep: Rand.Intn requires n > 0")
	}
	// Fast path for powers of two: no rejection needed, and scheduler choices
	// among 2, 4 or 8 actors are common enough to be worth it.
	if n&(n-1) == 0 {
		return int(r.Uint64() & uint64(n-1))
	}
	bound := uint64(n)
	hi, lo := bits.Mul64(r.Uint64(), bound)
	if lo < bound {
		// Rejection zone. Computing the threshold this way avoids a division in
		// the common case, which is the point of Lemire's method.
		threshold := (-bound) % bound
		for lo < threshold {
			hi, lo = bits.Mul64(r.Uint64(), bound)
		}
	}
	return int(hi)
}

// Int63n returns a uniformly distributed int64 in [0, n).
func (r *Rand) Int63n(n int64) int64 {
	if n <= 0 {
		panic("lockstep: Rand.Int63n requires n > 0")
	}
	return int64(r.Uint64() >> 1 % uint64(n))
}

// Float64 returns a value in [0, 1).
//
// Built from 53 bits — the exact mantissa width of a float64 — so every
// representable value in the range is reachable and none is favoured. The naive
// alternative, dividing a full uint64 by 2^64, loses the low bits to rounding
// and cannot produce some representable values at all.
func (r *Rand) Float64() float64 {
	return float64(r.Uint64()>>11) / (1 << 53)
}

// Bool returns true with probability p, where p is clamped to [0, 1].
//
// Clamping rather than panicking is deliberate: fault-injection configuration is
// often computed (a rate scaled by a multiplier), and a probability that lands
// at 1.0000000001 through floating-point arithmetic should mean "always", not
// "crash the test suite".
func (r *Rand) Bool(p float64) bool {
	switch {
	case p <= 0:
		return false
	case p >= 1:
		return true
	}
	return r.Float64() < p
}

// Pick returns a uniformly chosen element of s, and false if s is empty.
//
// Generic so callers need no type assertions, and returning ok rather than
// panicking because "choose a node to crash, if any are left alive" is a normal
// thing to ask and an empty set is a normal answer.
func Pick[T any](r *Rand, s []T) (T, bool) {
	var zero T
	if len(s) == 0 {
		return zero, false
	}
	return s[r.Intn(len(s))], true
}

// Shuffle permutes s in place using a Fisher-Yates shuffle.
//
// Iterating downward and swapping with an index in [0, i] is the correct form.
// The common upward variant that swaps with an index over the whole slice
// produces a distribution that is not uniform — it is the classic shuffle bug,
// and in a simulator it would quietly bias which orderings ever get explored.
func Shuffle[T any](r *Rand, s []T) {
	for i := len(s) - 1; i > 0; i-- {
		j := r.Intn(i + 1)
		s[i], s[j] = s[j], s[i]
	}
}

// Duration returns a duration uniformly distributed in [lo, hi].
//
// Inclusive of hi, because callers write Duration(0, timeout) expecting the
// boundary case — a message arriving exactly at the deadline — to be reachable.
// That boundary is where timeout bugs live, so excluding it would exclude the
// most interesting case in the range.
func (r *Rand) Duration(lo, hi int64) int64 {
	switch {
	case hi < lo:
		lo, hi = hi, lo
	case hi == lo:
		return lo
	}
	return lo + r.Int63n(hi-lo+1)
}
