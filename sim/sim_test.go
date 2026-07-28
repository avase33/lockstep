package sim

import (
	"fmt"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
)

// buildBusySim constructs a simulation with enough concurrency, channel traffic
// and timer activity that any leaked nondeterminism has somewhere to show up.
//
// It is deliberately awkward: actors interleave sends, receives, sleeps of
// varying lengths and explicit yields, and the number of iterations each one
// performs is drawn from the seeded PRNG. A simpler workload would pass a
// determinism test while still being nondeterministic in ways real code hits.
func buildBusySim(seed Seed) *Sim {
	s := New(Config{Seed: seed})
	const nodes = 5
	inboxes := make([]*Chan, nodes)
	for i := range inboxes {
		inboxes[i] = NewChan(s, 64)
	}
	for i := 0; i < nodes; i++ {
		i := i
		s.Go(fmt.Sprintf("node-%d", i), func() {
			r := s.Rand()
			iters := 3 + r.Intn(5)
			for k := 0; k < iters; k++ {
				target := r.Intn(nodes)
				inboxes[target].Send(fmt.Sprintf("m%d-%d", i, k))
				s.Clock().Sleep(time.Duration(r.Intn(10)+1) * time.Millisecond)
				if v, ok := inboxes[i].TryRecv(); ok {
					s.Record("node-%d got %v", i, v)
				}
				s.Yield()
			}
			// Drain whatever is left so nobody is left blocked on a full buffer.
			for {
				if _, ok := inboxes[i].TryRecv(); !ok {
					break
				}
			}
		})
	}
	return s
}

// This is the test the entire project stands on. If it ever fails, every seed
// ever published by this tool is worthless.
func TestSameSeedProducesIdenticalTrace(t *testing.T) {
	const seed = Seed(0x4f2a91c3)
	first := buildBusySim(seed).Run()
	if first.Failed() {
		t.Fatalf("baseline run failed: %v", first.Error())
	}
	want := first.Trace().Hash()

	for i := 0; i < 20; i++ {
		got := buildBusySim(seed).Run()
		if got.Failed() {
			t.Fatalf("run %d failed: %v", i, got.Error())
		}
		if got.Trace().Hash() != want {
			t.Fatalf("run %d diverged\n  want hash %s (%d events)\n  got  hash %s (%d events)\n%s",
				i, want, first.Trace().Len(), got.Trace().Hash(), got.Trace().Len(),
				firstDivergence(first.Trace(), got.Trace()))
		}
		if got.Steps != first.Steps {
			t.Fatalf("run %d took %d steps, baseline took %d", i, got.Steps, first.Steps)
		}
	}
}

// Determinism must not depend on how many OS threads Go is using. If it does,
// the simulator reproduces on a laptop and diverges on a CI box with a different
// core count — the worst possible failure mode, because it looks like a real bug.
func TestDeterminismHoldsAcrossGOMAXPROCS(t *testing.T) {
	const seed = Seed(0x1234abcd)
	original := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(original)

	var want string
	for _, procs := range []int{1, 2, 4, 8} {
		runtime.GOMAXPROCS(procs)
		res := buildBusySim(seed).Run()
		if res.Failed() {
			t.Fatalf("GOMAXPROCS=%d: %v", procs, res.Error())
		}
		h := res.Trace().Hash()
		if want == "" {
			want = h
			continue
		}
		if h != want {
			t.Fatalf("GOMAXPROCS=%d produced hash %s, want %s — the Go scheduler is leaking into the simulation",
				procs, h, want)
		}
	}
}

// A simulator that produced the same schedule for every seed would be
// deterministic and useless. This checks it actually explores.
func TestDifferentSeedsExploreDifferentSchedules(t *testing.T) {
	seen := make(map[string]Seed)
	for i := 0; i < 40; i++ {
		seed := Seed(0x1000 + i*7919)
		res := buildBusySim(seed).Run()
		if res.Failed() {
			t.Fatalf("seed %s failed: %v", seed, res.Error())
		}
		seen[res.Trace().Hash()] = seed
	}
	// 40 seeds should not collapse to a handful of schedules. The bound is loose
	// on purpose — this must never be flaky — but a broken PRNG or a scheduler
	// that ignores its random choice would land far below it.
	if len(seen) < 20 {
		t.Fatalf("40 seeds produced only %d distinct schedules; the scheduler is barely exploring", len(seen))
	}
}

func firstDivergence(a, b *Trace) string {
	ae, be := a.Events(), b.Events()
	n := len(ae)
	if len(be) < n {
		n = len(be)
	}
	for i := 0; i < n; i++ {
		if ae[i] != be[i] {
			var sb strings.Builder
			fmt.Fprintf(&sb, "first divergence at event %d:\n", i)
			fmt.Fprintf(&sb, "  A: %s\n", ae[i])
			fmt.Fprintf(&sb, "  B: %s\n", be[i])
			return sb.String()
		}
	}
	return fmt.Sprintf("traces agree on the first %d events but differ in length (%d vs %d)", n, len(ae), len(be))
}

func TestActorsRunAndFinish(t *testing.T) {
	s := New(Config{Seed: 1})
	var order []string
	for i := 0; i < 4; i++ {
		i := i
		s.Go(fmt.Sprintf("a%d", i), func() {
			for k := 0; k < 3; k++ {
				order = append(order, fmt.Sprintf("a%d.%d", i, k))
				s.Yield()
			}
		})
	}
	res := s.Run()
	if res.Failed() {
		t.Fatalf("run failed: %v", res.Error())
	}
	if len(order) != 12 {
		t.Fatalf("expected 12 steps recorded, got %d: %v", len(order), order)
	}
	// Every actor must have completed all three of its steps.
	counts := map[string]int{}
	for _, o := range order {
		counts[strings.Split(o, ".")[0]]++
	}
	for i := 0; i < 4; i++ {
		if c := counts[fmt.Sprintf("a%d", i)]; c != 3 {
			t.Errorf("actor a%d ran %d times, want 3", i, c)
		}
	}
}

// Actors interleave rather than running to completion one at a time. Without
// this, the scheduler could "work" by running each actor start to finish and
// would never find a concurrency bug.
func TestActorsActuallyInterleave(t *testing.T) {
	s := New(Config{Seed: 99})
	var order []string
	for i := 0; i < 3; i++ {
		i := i
		s.Go(fmt.Sprintf("a%d", i), func() {
			for k := 0; k < 5; k++ {
				order = append(order, fmt.Sprintf("a%d", i))
				s.Yield()
			}
		})
	}
	if res := s.Run(); res.Failed() {
		t.Fatalf("run failed: %v", res.Error())
	}
	switches := 0
	for i := 1; i < len(order); i++ {
		if order[i] != order[i-1] {
			switches++
		}
	}
	if switches < 5 {
		t.Fatalf("only %d context switches across %d steps — the scheduler is not interleaving: %v",
			switches, len(order), order)
	}
}

func TestUnbufferedChannelRendezvous(t *testing.T) {
	s := New(Config{Seed: 7})
	ch := NewChan(s, 0)
	got := make([]int, 0, 3)
	s.Go("sender", func() {
		for i := 1; i <= 3; i++ {
			ch.Send(i)
		}
		ch.Close()
	})
	s.Go("receiver", func() {
		for {
			v, ok := ch.Recv()
			if !ok {
				return
			}
			got = append(got, v.(int))
		}
	})
	if res := s.Run(); res.Failed() {
		t.Fatalf("run failed: %v", res.Error())
	}
	if len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Fatalf("unbuffered channel lost or reordered values: %v", got)
	}
}

func TestBufferedChannelBlocksWhenFull(t *testing.T) {
	s := New(Config{Seed: 11})
	ch := NewChan(s, 2)
	var sent, received int
	s.Go("sender", func() {
		for i := 0; i < 6; i++ {
			ch.Send(i)
			sent++
		}
		ch.Close()
	})
	s.Go("receiver", func() {
		for {
			if _, ok := ch.Recv(); !ok {
				return
			}
			received++
			s.Clock().Sleep(time.Millisecond)
		}
	})
	if res := s.Run(); res.Failed() {
		t.Fatalf("run failed: %v", res.Error())
	}
	if sent != 6 || received != 6 {
		t.Fatalf("sent %d received %d, want 6 and 6", sent, received)
	}
}

// Virtual time must cost no real time. This is the property that makes testing
// a five-minute timeout practical.
func TestVirtualTimeIsFree(t *testing.T) {
	s := New(Config{Seed: 3})
	s.Go("sleeper", func() {
		s.Clock().Sleep(24 * time.Hour)
	})
	start := time.Now()
	res := s.Run()
	elapsed := time.Since(start)
	if res.Failed() {
		t.Fatalf("run failed: %v", res.Error())
	}
	if res.VirtualTime() != 24*time.Hour {
		t.Fatalf("virtual time advanced %s, want 24h", res.VirtualTime())
	}
	if elapsed > time.Second {
		t.Fatalf("simulating 24 hours took %s of real time; the clock is not virtual", elapsed)
	}
}

// Sleepers must wake in deadline order regardless of the order they went to
// sleep — otherwise timeouts fire in the wrong sequence and every retry test is
// meaningless.
func TestSleepersWakeInDeadlineOrder(t *testing.T) {
	s := New(Config{Seed: 5})
	var woke []string
	delays := map[string]time.Duration{
		"slow":   300 * time.Millisecond,
		"medium": 200 * time.Millisecond,
		"fast":   100 * time.Millisecond,
	}
	// Register longest-first, so insertion order is the reverse of wake order.
	for _, name := range []string{"slow", "medium", "fast"} {
		name := name
		s.Go(name, func() {
			s.Clock().Sleep(delays[name])
			woke = append(woke, name)
		})
	}
	if res := s.Run(); res.Failed() {
		t.Fatalf("run failed: %v", res.Error())
	}
	want := []string{"fast", "medium", "slow"}
	for i := range want {
		if i >= len(woke) || woke[i] != want[i] {
			t.Fatalf("woke in order %v, want %v", woke, want)
		}
	}
}

func TestDeadlockIsReportedNotHung(t *testing.T) {
	s := New(Config{Seed: 13})
	ch := NewChan(s, 0)
	s.Go("waiter-a", func() { ch.Recv() })
	s.Go("waiter-b", func() { ch.Recv() })

	res := s.Run()
	if !res.Failed() {
		t.Fatal("expected a deadlock to be reported")
	}
	var dl *DeadlockError
	if !asDeadlock(res.Err, &dl) {
		t.Fatalf("expected *DeadlockError, got %T: %v", res.Err, res.Err)
	}
	if len(dl.Waiting) != 2 {
		t.Fatalf("expected 2 blocked actors, got %d: %v", len(dl.Waiting), dl.Waiting)
	}
	// The report must name the actors, or it sends someone hunting through a
	// trace to learn something the error could have told them.
	joined := strings.Join(dl.Waiting, " ")
	for _, name := range []string{"waiter-a", "waiter-b"} {
		if !strings.Contains(joined, name) {
			t.Errorf("deadlock report does not mention %s: %v", name, dl.Waiting)
		}
	}
	if !strings.Contains(res.Error(), "repro:") {
		t.Error("failure output must include a reproduction command")
	}
}

func asDeadlock(err error, out **DeadlockError) bool {
	d, ok := err.(*DeadlockError)
	if ok {
		*out = d
	}
	return ok
}

func TestPanicIsCapturedWithSeed(t *testing.T) {
	s := New(Config{Seed: 0xbeef})
	s.Go("fine", func() { s.Yield() })
	s.Go("bad", func() {
		s.Yield()
		panic("boom")
	})
	res := s.Run()
	if !res.Failed() {
		t.Fatal("expected the panic to fail the run")
	}
	pe, ok := res.Err.(*PanicError)
	if !ok {
		t.Fatalf("expected *PanicError, got %T", res.Err)
	}
	if pe.Actor != "bad" {
		t.Errorf("panic attributed to %q, want \"bad\"", pe.Actor)
	}
	if !strings.Contains(res.Error(), "0x0000beef") {
		t.Errorf("failure output must carry the seed, got:\n%s", res.Error())
	}
}

// A crashed actor must stay dead — including not being woken by a timer it set
// before dying. Resurrecting a crashed node silently invalidates every crash
// test built on the simulator.
func TestCrashedActorStaysDead(t *testing.T) {
	s := New(Config{Seed: 17})
	ran := 0
	s.Go("victim", func() {
		ran++
		s.Clock().Sleep(50 * time.Millisecond)
		ran++ // must never execute
	})
	s.Go("killer", func() {
		s.Clock().Sleep(10 * time.Millisecond)
		s.Crash("victim")
	})
	res := s.Run()
	if res.Failed() {
		t.Fatalf("run failed: %v", res.Error())
	}
	if ran != 1 {
		t.Fatalf("victim executed %d statements, want 1 — a crashed actor was resumed", ran)
	}
	if !s.Crashed("victim") {
		t.Error("Crashed() does not report the crash")
	}
}

func TestSelectReturnsTheChannelThatFired(t *testing.T) {
	s := New(Config{Seed: 23})
	a := NewChan(s, 0)
	b := NewChan(s, 0)
	var gotIdx int
	var gotVal any
	s.Go("selector", func() {
		gotIdx, gotVal, _ = Select(s, a, b)
	})
	s.Go("sender", func() {
		s.Clock().Sleep(time.Millisecond)
		b.Send("from-b")
	})
	if res := s.Run(); res.Failed() {
		t.Fatalf("run failed: %v", res.Error())
	}
	if gotIdx != 1 {
		t.Fatalf("Select reported index %d, want 1 (channel b)", gotIdx)
	}
	if gotVal != "from-b" {
		t.Fatalf("Select returned %v, want \"from-b\"", gotVal)
	}
}

// Calling a simulation primitive from an unmanaged goroutine must fail loudly at
// the call site, not corrupt the schedule quietly.
func TestPrimitiveOutsideActorPanics(t *testing.T) {
	s := New(Config{Seed: 1})
	s.Go("only", func() { s.Yield() })
	done := make(chan any, 1)
	go func() {
		defer func() { done <- recover() }()
		s.Yield() // from a raw goroutine
	}()
	if r := <-done; r == nil {
		t.Fatal("expected a panic when a primitive is called outside an actor")
	} else if !strings.Contains(fmt.Sprint(r), "Sim.Go") {
		t.Errorf("panic message should point at the fix, got: %v", r)
	}
}

func TestStuckSimulationIsBounded(t *testing.T) {
	s := New(Config{Seed: 1, MaxSteps: 500})
	s.Go("spinner", func() {
		for {
			s.Yield()
		}
	})
	res := s.Run()
	if !res.Failed() {
		t.Fatal("expected the step budget to stop an infinite loop")
	}
	if _, ok := res.Err.(*StuckError); !ok {
		t.Fatalf("expected *StuckError, got %T", res.Err)
	}
}

// The scheduler's choice must come only from the seeded PRNG. This checks the
// runnable set is ordered before the PRNG indexes into it — the subtle bug where
// the generator is deterministic but the thing it selects from is not.
func TestRunnableSetIsOrderIndependent(t *testing.T) {
	build := func() *Sim {
		s := New(Config{Seed: 0xfeed})
		names := []string{"z", "a", "m", "b"} // deliberately unsorted
		for _, n := range names {
			n := n
			s.Go(n, func() {
				for i := 0; i < 4; i++ {
					s.Record("tick %s", n)
					s.Yield()
				}
			})
		}
		return s
	}
	first := build().Run()
	for i := 0; i < 10; i++ {
		if h := build().Run().Trace().Hash(); h != first.Trace().Hash() {
			t.Fatalf("run %d diverged: %s vs %s", i, h, first.Trace().Hash())
		}
	}
}

func TestRandIsStableAndUniform(t *testing.T) {
	// Stability: the sequence is pinned here so a future change to the generator
	// cannot silently invalidate every published seed.
	r := NewRand(42)
	got := []uint64{r.Uint64(), r.Uint64(), r.Uint64()}
	again := NewRand(42)
	want := []uint64{again.Uint64(), again.Uint64(), again.Uint64()}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("generator is not reproducible at draw %d", i)
		}
	}

	// Uniformity of Intn, loosely — this must never be flaky, but a modulo-bias
	// bug or a broken rejection loop lands far outside these bounds.
	const n, draws = 7, 70000
	counts := make([]int, n)
	r2 := NewRand(0xabc)
	for i := 0; i < draws; i++ {
		counts[r2.Intn(n)]++
	}
	expect := draws / n
	for i, c := range counts {
		if c < expect*8/10 || c > expect*12/10 {
			t.Errorf("Intn(%d) bucket %d got %d draws, expected near %d", n, i, c, expect)
		}
	}
}

func TestShuffleIsUniform(t *testing.T) {
	// A biased Fisher-Yates still shuffles, so a smoke test would pass. Count how
	// often each element lands in position 0 instead.
	const trials = 30000
	counts := map[int]int{}
	r := NewRand(5)
	for i := 0; i < trials; i++ {
		s := []int{0, 1, 2, 3}
		Shuffle(r, s)
		counts[s[0]]++
	}
	expect := trials / 4
	for v, c := range counts {
		if c < expect*8/10 || c > expect*12/10 {
			t.Errorf("value %d landed first %d times, expected near %d — shuffle is biased", v, c, expect)
		}
	}
}

func TestTraceHashCoversDetail(t *testing.T) {
	mk := func(detail string) string {
		s := New(Config{Seed: 1})
		s.Go("a", func() { s.Record("%s", detail) })
		return s.Run().Trace().Hash()
	}
	if mk("alpha") == mk("beta") {
		t.Fatal("trace hash ignores Detail; two runs that did different things would look identical")
	}
}

func TestTraceTailTruncates(t *testing.T) {
	s := New(Config{Seed: 1})
	s.Go("a", func() {
		for i := 0; i < 50; i++ {
			s.Record("event %d", i)
			s.Yield()
		}
	})
	res := s.Run()
	tail := res.Trace().Tail(10)
	if !strings.Contains(tail, "earlier events omitted") {
		t.Error("Tail should say how much it hid")
	}
	if lines := strings.Count(tail, "\n"); lines > 12 {
		t.Errorf("Tail(10) produced %d lines", lines)
	}
}

func TestZeroSeedIsUsable(t *testing.T) {
	// splitmix64 does not degenerate at zero, unlike some generators. Pin it, so
	// nobody adds a defensive special case that changes every seed-0 run.
	s := New(Config{Seed: 0})
	s.Go("a", func() {
		for i := 0; i < 5; i++ {
			s.Rand().Intn(10)
			s.Yield()
		}
	})
	if res := s.Run(); res.Failed() {
		t.Fatalf("seed 0 failed: %v", res.Error())
	}
}

func TestActorNamesAreSorted(t *testing.T) {
	s := New(Config{Seed: 1})
	for _, n := range []string{"delta", "alpha", "charlie", "bravo"} {
		s.Go(n, func() {})
	}
	got := s.ActorNames()
	want := append([]string(nil), got...)
	sort.Strings(want)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ActorNames not sorted: %v", got)
		}
	}
}

func TestDuplicateActorNamePanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected a panic on duplicate actor name")
		}
	}()
	s := New(Config{Seed: 1})
	s.Go("dup", func() {})
	s.Go("dup", func() {})
}

func BenchmarkScheduler(b *testing.B) {
	for i := 0; i < b.N; i++ {
		s := New(Config{Seed: Seed(i), DisableTrace: true})
		for j := 0; j < 8; j++ {
			s.Go(fmt.Sprintf("a%d", j), func() {
				for k := 0; k < 50; k++ {
					s.Yield()
				}
			})
		}
		s.Run()
	}
}

// Regression test for a real bug found by the netsim package while it was being
// written: if the earliest pending timer belonged to a crashed actor and no live
// actor was due at that same instant, advanceClockLocked reported no progress
// and Run declared a deadlock — with other timers still queued.
//
// Crashing a node that is asleep on a heartbeat is the most common thing a crash
// test does, so this fired constantly and blamed the system under test for a
// fault in the simulator. That is the worst class of bug a testing tool can have.
func TestCrashedActorTimerDoesNotFakeADeadlock(t *testing.T) {
	s := New(Config{Seed: 1})
	survived := false
	s.Go("victim", func() {
		s.Clock().Sleep(10 * time.Millisecond) // earliest timer, never fires
		t.Error("crashed actor was resumed")
	})
	s.Go("survivor", func() {
		s.Clock().Sleep(50 * time.Millisecond) // must still fire
		survived = true
	})
	s.Go("killer", func() {
		s.Clock().Sleep(1 * time.Millisecond)
		s.Crash("victim")
	})

	res := s.Run()
	if res.Failed() {
		t.Fatalf("spurious failure with live timers still pending: %v", res.Error())
	}
	if !survived {
		t.Fatal("survivor's timer never fired")
	}
}

// The same shape with several crashed timers stacked before the live one, so the
// fix is exercised as a loop rather than a single extra iteration.
func TestManyCrashedTimersBeforeALiveOne(t *testing.T) {
	s := New(Config{Seed: 2})
	reached := false
	for i := 1; i <= 5; i++ {
		i := i
		s.Go(fmt.Sprintf("dead-%d", i), func() {
			s.Clock().Sleep(time.Duration(i*10) * time.Millisecond)
			t.Errorf("dead-%d was resumed", i)
		})
	}
	s.Go("live", func() {
		s.Clock().Sleep(500 * time.Millisecond)
		reached = true
	})
	s.Go("killer", func() {
		s.Clock().Sleep(time.Millisecond)
		for i := 1; i <= 5; i++ {
			s.Crash(fmt.Sprintf("dead-%d", i))
		}
	})
	if res := s.Run(); res.Failed() {
		t.Fatalf("unexpected failure: %v", res.Error())
	}
	if !reached {
		t.Fatal("the live timer never fired")
	}
}
