package netsim

import (
	"runtime"
	"testing"
	"time"

	"github.com/avase33/lockstep/sim"
)

// ping is a message body. Deliberately a plain value type: a pointer body would
// be shared between sender and receiver, which is not what a real wire does.
type ping struct {
	From  string
	Round int
}

// buildChaosSim wires a small cluster on a network that is misbehaving in every
// way this package knows how to misbehave, and returns it ready to Run.
//
// Every axis is switched on at once — loss, duplication, a wide latency range,
// a symmetric partition, a one-way partition, a heal, and a crash — because a
// determinism test is only as good as the nondeterminism it gives leaked
// randomness somewhere to hide. A workload that only sent messages would pass
// while a partition helper iterated a map in whatever order Go felt like.
//
// It also has to terminate. The sending nodes poll with TryRecv and sleep rather
// than blocking in Recv, so none of them is left parked when its peers finish,
// and couriers still in flight at the end either deliver into a buffer or drop
// on overflow. A determinism test that deadlocks tells you nothing about
// determinism.
//
// The crash victim is "sink", the one node that blocks in Recv. That started as
// a workaround: sim's clock used to discard a crashed actor's pending timer and
// then report that it had woken nobody, so if that timer was the only one due at
// that instant, Run mistook the situation for a deadlock while later timers were
// still queued. Writing this package is what surfaced it.
//
// advanceClockLocked now loops past dead timers and the bug is fixed, with
// regression coverage in sim's TestCrashedActorTimerDoesNotFakeADeadlock. The
// choice of victim is kept because a node parked in Recv is the more interesting
// crash to model here — it dies holding an unprocessed inbox — not because the
// alternative is still broken. The kvstore example crashes replicas mid-Sleep
// routinely and does not trip it.
func buildChaosSim(seed sim.Seed) (*sim.Sim, *Network) {
	s := sim.New(sim.Config{Seed: seed})
	net := New(s, Config{
		MinLatency:    1 * time.Millisecond,
		MaxLatency:    25 * time.Millisecond,
		DropRate:      0.12,
		DuplicateRate: 0.15,
		InboxCapacity: 32,
	})

	names := []string{"n0", "n1", "n2", "n3", "sink"}
	for _, name := range names[:len(names)-1] {
		node := net.Join(name)
		s.Go(name, func() {
			r := s.Rand()
			rounds := 6 + r.Intn(6)
			for k := 0; k < rounds; k++ {
				node.Send(names[r.Intn(len(names))], ping{From: name, Round: k})
				for {
					m, ok := node.TryRecv()
					if !ok {
						break
					}
					s.Record("app %s <- %s #%d.%d", name, m.From, m.Seq, m.Copy)
				}
				s.Clock().Sleep(time.Duration(1+r.Intn(9)) * time.Millisecond)
			}
		})
	}

	// sink blocks in Recv, which is how a real server waits, and is the node the
	// chaos actor kills. Its inbox is deliberately the smallest, so overflow
	// drops get exercised too.
	sink := net.Join("sink")
	s.Go("sink", func() {
		for {
			m, ok := sink.Recv()
			if !ok {
				return
			}
			s.Record("app sink <- %s #%d.%d", m.From, m.Seq, m.Copy)
		}
	})

	s.Go("chaos", func() {
		r := s.Rand()
		s.Clock().Sleep(time.Duration(5+r.Intn(10)) * time.Millisecond)
		net.Partition([]string{"n0", "n1"}, []string{"n2", "n3", "sink"})
		s.Clock().Sleep(time.Duration(5+r.Intn(15)) * time.Millisecond)
		net.PartitionOneWay([]string{"sink"}, []string{"n0", "n1"})
		s.Clock().Sleep(time.Duration(5+r.Intn(15)) * time.Millisecond)
		net.Heal()
		s.Clock().Sleep(time.Duration(1+r.Intn(10)) * time.Millisecond)
		s.Crash("sink")
	})

	return s, net
}

// This is the test the package stands on. Everything else here checks that a
// particular fault behaves; this checks that any of it is worth trusting.
//
// Twenty runs rather than two, because a determinism bug sourced from Go's map
// randomisation or from the runtime scheduler does not fail every time — it
// fails at whatever rate the runtime happens to reorder things, and a two-run
// test would report green on a network that diverges one time in ten.
func TestSameSeedProducesIdenticalTrace(t *testing.T) {
	const seed = sim.Seed(0x9c1f04ab)

	s, net := buildChaosSim(seed)
	first := s.Run()
	if first.Failed() {
		t.Fatalf("baseline run failed: %v", first.Error())
	}
	want := first.Trace().Hash()
	wantStats := net.Stats()

	// A trace of a workload that did nothing would also be perfectly
	// reproducible, so assert the workload actually exercised the faults before
	// trusting the hash. This guard is not decoration: an earlier version of this
	// test passed with DropRate misspelt into a field that no longer existed.
	if wantStats.Sent == 0 || wantStats.Dropped == 0 || wantStats.Duplicated == 0 || wantStats.Delivered == 0 {
		t.Fatalf("chaos workload did not exercise the fault injector: %s", wantStats)
	}
	if first.Trace().Len() < 100 {
		t.Fatalf("trace too short to be meaningful: %d events", first.Trace().Len())
	}

	for i := 0; i < 20; i++ {
		s, net := buildChaosSim(seed)
		got := s.Run()
		if got.Failed() {
			t.Fatalf("run %d failed: %v", i, got.Error())
		}
		if h := got.Trace().Hash(); h != want {
			t.Fatalf("run %d diverged\n  want hash %s (%d events)\n  got  hash %s (%d events)\n  want stats %s\n  got  stats %s",
				i, want, first.Trace().Len(), h, got.Trace().Len(), wantStats, net.Stats())
		}
		if got.Steps != first.Steps {
			t.Fatalf("run %d took %d steps, baseline took %d", i, got.Steps, first.Steps)
		}
		if net.Stats() != wantStats {
			t.Fatalf("run %d stats %s, baseline %s", i, net.Stats(), wantStats)
		}
	}
}

// Determinism must not depend on how many OS threads Go is using, or the
// simulator reproduces on a laptop and diverges on a CI box with a different
// core count — the worst failure mode available, because divergence looks
// exactly like a real bug in the system under test.
func TestDeterminismHoldsAcrossGOMAXPROCS(t *testing.T) {
	orig := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(orig)

	const seed = sim.Seed(0x2b7e1516)
	var want string
	for _, procs := range []int{1, 2, 4, 8} {
		runtime.GOMAXPROCS(procs)
		s, _ := buildChaosSim(seed)
		res := s.Run()
		if res.Failed() {
			t.Fatalf("GOMAXPROCS=%d: run failed: %v", procs, res.Error())
		}
		got := res.Trace().Hash()
		if want == "" {
			want = got
			continue
		}
		if got != want {
			t.Fatalf("GOMAXPROCS=%d produced trace hash %s, want %s", procs, got, want)
		}
	}
}

// Reordering is the fault this package gets for free from per-copy latency, and
// "for free" is exactly the sort of claim that turns out to be false. Assert it.
//
// The sender emits an ascending sequence in one uninterrupted burst — Send never
// yields, so all of it leaves at the same virtual instant and any difference in
// arrival is attributable to the network.
//
// Two things are asserted, and the second exists because the first is not enough.
// Counting inversions in the delivery order proves messages arrive out of order,
// but it would still pass on a network with a fixed latency: every courier would
// then wake at the same instant and sim's scheduler, which chooses among
// simultaneously runnable actors at random, would shuffle them anyway. That is a
// real reordering, but it is the scheduler's, not the network's. The second
// assertion counts overtakes in *virtual time* — a later-sent message reaching
// the receiver at a strictly earlier instant — which only independent per-copy
// latency draws can produce. An earlier version of this test had only the first
// assertion and happily passed a mutant whose latency was constant.
func TestLatencyReordersMessages(t *testing.T) {
	const count = 24

	type arrival struct {
		order int // position in send order
		at    time.Time
	}

	inversions, overtakes := 0, 0
	for _, seed := range []sim.Seed{0x01, 0x02, 0x03, 0x04, 0x05} {
		s := sim.New(sim.Config{Seed: seed})
		net := New(s, Config{
			MinLatency:    1 * time.Millisecond,
			MaxLatency:    200 * time.Millisecond,
			InboxCapacity: count + 1,
		})
		a := net.Join("a")
		b := net.Join("b")

		var got []arrival
		s.Go("a", func() {
			for i := 0; i < count; i++ {
				a.Send("b", i)
			}
		})
		s.Go("b", func() {
			for i := 0; i < count; i++ {
				m, ok := b.Recv()
				if !ok {
					return
				}
				got = append(got, arrival{order: m.Body.(int), at: s.Clock().Now()})
			}
		})

		res := s.Run()
		if res.Failed() {
			t.Fatalf("seed %s: run failed: %v", seed, res.Error())
		}
		if len(got) != count {
			t.Fatalf("seed %s: received %d of %d messages (no faults were configured)", seed, len(got), count)
		}
		for i := 1; i < len(got); i++ {
			if got[i].order < got[i-1].order {
				inversions++
			}
		}
		for i := range got {
			for j := range got {
				if got[j].order > got[i].order && got[j].at.Before(got[i].at) {
					overtakes++
				}
			}
		}
	}

	if inversions == 0 {
		t.Errorf("no message was ever delivered out of send order across %d messages and 5 seeds", count)
	}
	if overtakes == 0 {
		t.Errorf("no message ever arrived at a strictly earlier virtual time than one sent before it; "+
			"latency is not being drawn independently per message (%d messages, 5 seeds)", count)
	}
	t.Logf("observed %d out-of-order arrivals and %d virtual-time overtakes", inversions, overtakes)
}

// The drop rate has to mean something. If the configured probability and the
// observed frequency disagree, every conclusion drawn from a "10% loss" run is
// wrong in a direction nobody will check.
//
// Bounds are deliberately loose. This test is seeded and therefore not actually
// random, but a tight bound would still turn any future change to the order of
// PRNG draws into a mysterious statistical failure, and a test that fails for
// reasons unrelated to its name is worse than no test.
func TestDropRateMatchesConfiguration(t *testing.T) {
	const (
		perRun    = 1500
		rate      = 0.30
		tolerance = 0.05
	)

	var sent, dropped, delivered uint64
	for _, seed := range []sim.Seed{0x11, 0x22} {
		s := sim.New(sim.Config{Seed: seed})
		net := New(s, Config{DropRate: rate, InboxCapacity: perRun + 1})
		a := net.Join("a")
		net.Join("b")

		s.Go("a", func() {
			for i := 0; i < perRun; i++ {
				a.Send("b", i)
				// Breathe every so often so in-flight couriers retire instead of
				// piling up. Without this the scheduler carries a few thousand
				// runnable actors and the test gets slow enough that people stop
				// running it, which is its own kind of failure.
				if i%50 == 49 {
					s.Clock().Sleep(time.Millisecond)
				}
			}
		})

		res := s.Run()
		if res.Failed() {
			t.Fatalf("seed %s: run failed: %v", seed, res.Error())
		}
		st := net.Stats()
		sent += st.Sent
		dropped += st.Dropped
		delivered += st.Delivered
		// "b" has no reader, so its inbox was sized to hold everything. If it
		// had overflowed, overflow drops would be counted as loss and the
		// measurement below would be silently wrong.
		if st.Delivered+st.Dropped != st.Sent {
			t.Fatalf("seed %s: accounting broken, %s", seed, st)
		}
	}

	observed := float64(dropped) / float64(sent)
	if observed < rate-tolerance || observed > rate+tolerance {
		t.Fatalf("dropped %d of %d messages (%.4f), want within %.2f of %.2f",
			dropped, sent, observed, tolerance, rate)
	}
	t.Logf("observed drop rate %.4f over %d messages (delivered %d)", observed, sent, delivered)
}

// A duplicate that arrives once is not a duplicate. Assert both copies land, and
// assert they are distinguishable, because a network that delivered the same
// copy twice would pass a naive count-only check.
func TestDuplicatesAreDeliveredTwice(t *testing.T) {
	const count = 8

	s := sim.New(sim.Config{Seed: 0x0d00d})
	net := New(s, Config{
		DuplicateRate: 1.0,
		MinLatency:    1 * time.Millisecond,
		MaxLatency:    60 * time.Millisecond,
		InboxCapacity: 4 * count,
	})
	a := net.Join("a")
	b := net.Join("b")

	arrivals := make([]int, count+1)   // indexed by Seq
	copiesSeen := make([]int, count+1) // bitmask of Copy values per Seq
	overtakes := 0                     // times the duplicate beat the original
	firstCopy := make([]int, count+1)

	s.Go("a", func() {
		for i := 0; i < count; i++ {
			a.Send("b", i)
		}
	})
	s.Go("b", func() {
		for i := 0; i < 2*count; i++ {
			m, ok := b.Recv()
			if !ok {
				return
			}
			arrivals[m.Seq]++
			copiesSeen[m.Seq] |= 1 << m.Copy
			if firstCopy[m.Seq] == 0 {
				firstCopy[m.Seq] = m.Copy
			}
		}
	})

	res := s.Run()
	if res.Failed() {
		t.Fatalf("run failed: %v", res.Error())
	}

	for seq := 1; seq <= count; seq++ {
		if arrivals[seq] != 2 {
			t.Errorf("message #%d arrived %d times, want 2", seq, arrivals[seq])
		}
		if copiesSeen[seq] != (1<<1)|(1<<2) {
			t.Errorf("message #%d arrived as copies mask %b, want both copy 1 and copy 2", seq, copiesSeen[seq])
		}
		if firstCopy[seq] == 2 {
			overtakes++
		}
	}

	st := net.Stats()
	if st.Sent != count || st.Duplicated != count || st.Delivered != 2*count || st.Dropped != 0 {
		t.Fatalf("unexpected counters: %s", st)
	}
	// Not asserted, only reported: the duplicate drawing its own latency means it
	// sometimes arrives first, which is the case that breaks at-most-once
	// handling. Asserting a minimum here would bind the test to one seed's luck.
	t.Logf("%d of %d duplicates overtook their original", overtakes, count)
}

// phase is one observation in the partition tests: a single driver actor sends
// on behalf of both endpoints, waits out the wire time, and records what landed.
// One driver rather than two node actors because the phases have to happen in a
// known order, and coordinating two actors to agree on "now" would test the
// coordination rather than the partition.
//
// Neither endpoint has an actor of its own. That is legal — the network only
// consults Sim.Crashed for a node's name — and it keeps the test to the one
// thing it is about.
//
// Results are collected into a slice and asserted after Run rather than with
// t.Fatalf inside the actor, because testing.T.Fatal calls runtime.Goexit and
// must run on the test's own goroutine; calling it from an actor would abandon
// the simulation mid-hand-off instead of failing the test.
type phase struct {
	label   string
	aToB    int
	bToA    int
	blocked [2]bool // Blocked(a,b), Blocked(b,a)
}

func TestPartitionBlocksBothWaysAndHealRestores(t *testing.T) {
	const wire = 5 * time.Millisecond

	s := sim.New(sim.Config{Seed: 0x7a11})
	net := New(s, Config{MinLatency: wire, MaxLatency: wire, InboxCapacity: 16})
	a := net.Join("a")
	b := net.Join("b")

	var phases []phase
	s.Go("driver", func() {
		probe := func(label string) {
			net.Send("a", "b", label)
			net.Send("b", "a", label)
			s.Clock().Sleep(4 * wire)
			p := phase{label: label, aToB: b.Pending(), bToA: a.Pending()}
			p.blocked[0], p.blocked[1] = net.Blocked("a", "b"), net.Blocked("b", "a")
			phases = append(phases, p)
			for _, ok := b.TryRecv(); ok; _, ok = b.TryRecv() {
			}
			for _, ok := a.TryRecv(); ok; _, ok = a.TryRecv() {
			}
		}
		probe("healthy")
		net.Partition([]string{"a"}, []string{"b"})
		probe("partitioned")
		net.Heal()
		probe("healed")
	})

	res := s.Run()
	if res.Failed() {
		t.Fatalf("run failed: %v", res.Error())
	}
	if len(phases) != 3 {
		t.Fatalf("driver recorded %d phases, want 3", len(phases))
	}

	want := []phase{
		{label: "healthy", aToB: 1, bToA: 1, blocked: [2]bool{false, false}},
		{label: "partitioned", aToB: 0, bToA: 0, blocked: [2]bool{true, true}},
		{label: "healed", aToB: 1, bToA: 1, blocked: [2]bool{false, false}},
	}
	for i, w := range want {
		if phases[i] != w {
			t.Errorf("phase %q: got %+v, want %+v", w.label, phases[i], w)
		}
	}
}

// The asymmetric case, asserted in both directions in the same run.
//
// Checking only the blocked direction would pass on a network that had simply
// stopped delivering anything, and checking only the open direction would pass
// on a network that ignored partitions entirely. The bug this catches is a
// partition implementation that stores undirected pairs, which looks correct
// until a consensus test starts electing two leaders.
func TestAsymmetricPartitionBlocksOneDirectionOnly(t *testing.T) {
	const wire = 5 * time.Millisecond

	s := sim.New(sim.Config{Seed: 0xa5ee})
	net := New(s, Config{MinLatency: wire, MaxLatency: wire, InboxCapacity: 16})
	a := net.Join("a")
	b := net.Join("b")

	var gotAtB, gotAtA int
	var blockedAB, blockedBA bool
	s.Go("driver", func() {
		// A can reach B, but B cannot reach A: block only the b->a direction.
		net.PartitionOneWay([]string{"b"}, []string{"a"})
		blockedAB, blockedBA = net.Blocked("a", "b"), net.Blocked("b", "a")

		net.Send("a", "b", "forward")
		net.Send("b", "a", "reverse")
		s.Clock().Sleep(4 * wire)
		gotAtB, gotAtA = b.Pending(), a.Pending()
	})

	res := s.Run()
	if res.Failed() {
		t.Fatalf("run failed: %v", res.Error())
	}
	if blockedAB {
		t.Errorf("Blocked(a,b) = true; the a->b direction should be open")
	}
	if !blockedBA {
		t.Errorf("Blocked(b,a) = false; the b->a direction should be cut")
	}
	if gotAtB != 1 {
		t.Errorf("b received %d messages from a, want 1 (that direction is open)", gotAtB)
	}
	if gotAtA != 0 {
		t.Errorf("a received %d messages from b, want 0 (that direction is cut)", gotAtA)
	}
	if st := net.Stats(); st.Sent != 2 || st.Delivered != 1 || st.Dropped != 1 {
		t.Errorf("unexpected counters: %s", st)
	}
}

// A crash has to stop delivery, including for messages already on the wire when
// it happened — that is the case retry logic gets wrong.
//
// The healthy third node is the control. Without it, a network that had broken
// entirely would pass this test by delivering nothing at all, which is the
// classic way an assertion of absence proves nothing.
func TestMessageToCrashedNodeIsNotDelivered(t *testing.T) {
	const wire = 10 * time.Millisecond

	s := sim.New(sim.Config{Seed: 0xdead10cc})
	net := New(s, Config{MinLatency: wire, MaxLatency: wire, InboxCapacity: 16})
	a := net.Join("a")
	b := net.Join("b")
	c := net.Join("c")

	var atB, atC int
	s.Go("b", func() {
		for {
			if _, ok := b.Recv(); !ok {
				return
			}
			atB++
		}
	})
	s.Go("c", func() {
		for {
			if _, ok := c.Recv(); !ok {
				return
			}
			atC++
		}
	})
	s.Go("a", func() {
		a.Send("b", "in-flight-when-it-dies")
		a.Send("c", "control")
		s.Clock().Sleep(wire / 2) // crash while the first message is still travelling
		s.Crash("b")
		a.Send("b", "sent-to-a-corpse")
		s.Clock().Sleep(4 * wire)
		net.Close()
	})

	res := s.Run()
	if res.Failed() {
		t.Fatalf("run failed: %v", res.Error())
	}
	if atB != 0 {
		t.Errorf("crashed node received %d messages, want 0", atB)
	}
	if atC != 1 {
		t.Errorf("healthy control node received %d messages, want 1; the network is broken, not the crash handling", atC)
	}
	st := net.Stats()
	if st.Sent != 3 || st.Delivered != 1 || st.Dropped != 2 {
		t.Errorf("unexpected counters: %s", st)
	}
}

// The counters are the only summary anyone reads. If they do not add up, every
// statistical claim made from them is arithmetic on noise.
//
// The identity is Sent == Delivered + Dropped - Duplicated, which holds because
// each Send makes one copy (two when duplication fires) and each copy ends in
// exactly one of delivered or dropped.
func TestCountersBalance(t *testing.T) {
	for _, seed := range []sim.Seed{0x31, 0x32, 0x33, 0x34} {
		s, net := buildChaosSim(seed)
		res := s.Run()
		if res.Failed() {
			t.Fatalf("seed %s: run failed: %v", seed, res.Error())
		}
		st := net.Stats()

		// Signed arithmetic on purpose: the identity is stated with a
		// subtraction, and computing it in uint64 would wrap on a mismatch and
		// print a number in the quintillions instead of a small difference.
		lhs := int64(st.Sent)
		rhs := int64(st.Delivered) + int64(st.Dropped) - int64(st.Duplicated)
		if lhs != rhs {
			t.Errorf("seed %s: sent=%d but delivered+dropped-duplicated=%d (%s)", seed, lhs, rhs, st)
		}
		if !st.Balanced() {
			t.Errorf("seed %s: Stats.Balanced() disagrees with the identity: %s", seed, st)
		}
		if st.Sent == 0 || st.Duplicated == 0 || st.Dropped == 0 {
			t.Errorf("seed %s: workload did not exercise every counter: %s", seed, st)
		}
		if st.DelayedTotal <= 0 {
			t.Errorf("seed %s: no wire time accumulated: %s", seed, st)
		}
	}
}

// Causality. A simulated network may do anything except deliver a message before
// it was sent, because a violation there invalidates every ordering conclusion
// drawn from a run — and it is a plausible bug, since latency is a signed
// duration and virtual time is arithmetic on it.
//
// Both halves are checked: never earlier than the send (the invariant), and
// never at the same instant when MinLatency is positive (that the configured
// floor is actually applied, not silently rounded away).
func TestNoMessageIsDeliveredBeforeItWasSent(t *testing.T) {
	const minWire = 2 * time.Millisecond

	type violation struct {
		seq       uint64
		copyNo    int
		sentAt    time.Time
		recvAt    time.Time
		tooEarly  bool
		instantly bool
	}

	for _, seed := range []sim.Seed{0x41, 0x42, 0x43} {
		s := sim.New(sim.Config{Seed: seed})
		net := New(s, Config{
			MinLatency:    minWire,
			MaxLatency:    40 * time.Millisecond,
			DropRate:      0.1,
			DuplicateRate: 0.2,
			InboxCapacity: 64,
		})

		names := []string{"p", "q", "r"}
		var bad []violation
		received := 0
		for _, name := range names {
			node := net.Join(name)
			s.Go(name, func() {
				r := s.Rand()
				for k := 0; k < 10; k++ {
					node.Send(names[r.Intn(len(names))], k)
					for {
						m, ok := node.TryRecv()
						if !ok {
							break
						}
						received++
						now := s.Clock().Now()
						v := violation{
							seq: m.Seq, copyNo: m.Copy, sentAt: m.SentAt, recvAt: now,
							tooEarly:  now.Before(m.SentAt),
							instantly: !now.After(m.SentAt),
						}
						if v.tooEarly || v.instantly {
							bad = append(bad, v)
						}
					}
					s.Clock().Sleep(time.Duration(1+r.Intn(5)) * time.Millisecond)
				}
			})
		}

		res := s.Run()
		if res.Failed() {
			t.Fatalf("seed %s: run failed: %v", seed, res.Error())
		}
		if received == 0 {
			t.Fatalf("seed %s: nothing was delivered, so causality was not tested", seed)
		}
		for _, v := range bad {
			if v.tooEarly {
				t.Errorf("seed %s: message #%d.%d delivered at %s, before it was sent at %s",
					seed, v.seq, v.copyNo, v.recvAt.Sub(sim.DefaultStartTime), v.sentAt.Sub(sim.DefaultStartTime))
			} else {
				t.Errorf("seed %s: message #%d.%d delivered at the instant it was sent (%s) despite MinLatency=%s",
					seed, v.seq, v.copyNo, v.recvAt.Sub(sim.DefaultStartTime), minWire)
			}
		}
	}
}

// Configuration that cannot be honoured has to be repaired predictably, because
// fault rates are usually computed rather than written literally, and a run that
// panics on a scaled rate of 1.0000001 wastes an afternoon.
func TestConfigNormalisation(t *testing.T) {
	s := sim.New(sim.Config{Seed: 1})
	got := New(s, Config{MinLatency: 50 * time.Millisecond, MaxLatency: 10 * time.Millisecond}).Config()
	if got.MinLatency != 10*time.Millisecond || got.MaxLatency != 50*time.Millisecond {
		t.Errorf("inverted latency range was not normalised: %+v", got)
	}
	if got.InboxCapacity != DefaultInboxCapacity {
		t.Errorf("InboxCapacity = %d, want the default %d", got.InboxCapacity, DefaultInboxCapacity)
	}

	got = New(s, Config{MinLatency: -time.Second, MaxLatency: -time.Second}).Config()
	if got.MinLatency < 0 || got.MaxLatency < 0 {
		t.Errorf("negative latency survived normalisation: %+v", got)
	}
}
