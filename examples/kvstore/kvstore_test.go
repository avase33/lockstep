package kvstore

import (
	"strings"
	"testing"

	"github.com/avase33/lockstep/linz"
	"github.com/avase33/lockstep/sim"
)

// firstSeed is where every sweep in this file starts.
//
// Fixed rather than drawn from the clock, because the whole promise of the
// framework is that a run named by its seed can be rerun. A test that started
// somewhere new each time would be a test whose failures nobody could reproduce
// — which is the exact disease lockstep exists to cure, reintroduced in its own
// example.
const firstSeed = sim.Seed(1)

// sweepLimit is how many seeds a search for the bug is allowed. Generous
// relative to how many it actually takes, so that a change which makes the bug
// rarer shows up as a longer search rather than as a flaky test.
const sweepLimit = 64

// TestSingleReplicaReadsAreNotLinearizable is the demonstration.
//
// It sweeps seeds against the buggy store until one produces a history no
// correct key-value store could have produced, and then prints the seed, the
// checker's report and the command that replays it. The test PASSES when the
// violation is found: what is under test here is lockstep's ability to find a
// real bug and hand back a reproduction, not the store, which is known to be
// broken.
//
// Run it with -v to see the report.
func TestSingleReplicaReadsAreNotLinearizable(t *testing.T) {
	opt := DefaultOptions(SingleReplicaReads)

	out, tried, found := FirstViolation(firstSeed, sweepLimit, opt)
	if out.SimErr != nil {
		t.Fatalf("seed %s: the simulation itself failed, which is a bug in this example "+
			"or in lockstep rather than a finding about the store:\n%v", out.Seed, out.SimErr)
	}
	if !found {
		t.Fatalf("no linearizability violation in %d seeds starting at %s.\n"+
			"The bug is real, so this means the workload stopped exercising it: check that "+
			"faults are still being injected (%v) and that operations are still returning "+
			"(%d returned, %d never did).",
			tried, firstSeed, out.Faults, out.Returned, out.Pending)
	}

	// A violation found on a run where nothing was observed would be a violation
	// of nothing. Guard the guard.
	if out.Returned == 0 {
		t.Fatalf("seed %s violated, but no operation ever returned; the history is vacuous", out.Seed)
	}
	assertFaultsFired(t, out)

	t.Logf("\n"+
		"FAIL: linearizability violated by single-replica reads\n"+
		"  seed:   %s   (found after %d seed(s))\n"+
		"  trace:  %s\n"+
		"  ops:    %d returned, %d never returned, over %d keys\n"+
		"  net:    %s\n"+
		"  faults: %s\n"+
		"  repro:  %s\n\n"+
		"%v\n",
		out.Seed, tried, out.TraceHash,
		out.Returned, out.Pending, out.Check.Partitions,
		out.Net, strings.Join(out.Faults, "; "), out.Repro(),
		out.Check)

	// The checker names the operation that cannot have happened; this names the
	// replica that made it happen. Together they are the diagnosis, and printing
	// both is the difference between "the store is wrong" and "replica-1 missed
	// the write while it was partitioned and answered a read with what it had".
	if key := out.FailingKey(); key != "" {
		t.Logf("what the replicas did to key %q:\n%s\n", key, strings.Join(out.Story(key), "\n"))
	}
}

// TestQuorumReadsAreLinearizable is the test that makes the previous one worth
// anything.
//
// A checker that rejects everything has found nothing, and a fault injector
// violent enough to break any store is measuring itself rather than the system.
// So the fixed read path faces the identical workload — same clients, same
// keys, same faults, same seeds — and must survive every one of them.
//
// It also asserts that the runs were not vacuous. A workload where every
// operation timed out would be trivially linearizable, and would keep this test
// green while it quietly stopped testing anything.
func TestQuorumReadsAreLinearizable(t *testing.T) {
	// The seed count is the strength of the claim, so it is the one thing worth
	// spending time on: each seed is a different schedule, and a read path that
	// is wrong is wrong on a few percent of them. Under -race a run costs about
	// 170 ms, so -short trades most of the confidence for most of the time.
	seeds := 60
	if testing.Short() {
		seeds = 12
	}

	opt := DefaultOptions(QuorumReads)

	var returned, pending, checked int
	var first Outcome
	Sweep(firstSeed, seeds, opt, func(out Outcome) bool {
		checked++
		if checked == 1 {
			first = out
		}
		if out.SimErr != nil {
			t.Errorf("seed %s: simulation failed: %v", out.Seed, out.SimErr)
			return false
		}
		switch out.Check.Status {
		case linz.NotLinearizable:
			t.Errorf("seed %s: the CORRECT store was reported as violating linearizability.\n"+
				"Either the fix is not a fix, or the harness is crying wolf; both are worse than "+
				"the bug this example is about.\n  repro: %s\n\n%v",
				out.Seed, out.Repro(), out.Check)
			return false
		case linz.Unknown:
			t.Errorf("seed %s: the checker ran out of budget (widest partition %d operations).\n"+
				"Unknown is not a pass. Reduce Options.Clients, or raise the budget.\n%s",
				out.Seed, out.Check.Widest, out.Check.Reason)
			return false
		}
		returned += out.Returned
		pending += out.Pending
		return true
	})

	if checked != seeds {
		return // an error was already reported
	}
	if returned == 0 {
		t.Fatalf("%d seeds produced no operation that ever returned; a history nobody observed "+
			"is linearizable for the wrong reason", seeds)
	}
	// Most operations must complete, or the clean verdict is mostly a statement
	// about the timeouts. Two thirds is well below what the tuned workload
	// achieves and well above what a broken one would.
	if 3*returned < 2*(returned+pending) {
		t.Fatalf("only %d of %d operations returned across %d seeds; the workload is timing out "+
			"rather than exercising the store", returned, returned+pending, seeds)
	}
	assertFaultsFired(t, first)

	t.Logf("%d seeds, %s: all linearizable; %d operations returned, %d never did",
		seeds, QuorumReads, returned, pending)
}

// TestReproducibleFromSeed is the product promise, stated as an assertion.
//
// A seed that found a bug has to find the same bug, by the same route, every
// time — on this machine, on a colleague's, and in CI six months from now.
// "The test failed again" is not reproduction; it could be a different
// interleaving of the same defect, and fixing one would leave the other. So this
// compares the trace hash, which covers every scheduling decision the simulation
// made, alongside the checker's report.
//
// Ten replays rather than two, because a determinism bug sourced from Go's map
// randomisation does not fail every time. It fails at whatever rate the runtime
// happens to reorder things, and a two-run test reports green on a harness that
// diverges one run in ten.
func TestReproducibleFromSeed(t *testing.T) {
	const replays = 10

	opt := DefaultOptions(SingleReplicaReads)

	found, tried, ok := FirstViolation(firstSeed, sweepLimit, opt)
	if !ok {
		t.Fatalf("no violation in %d seeds; nothing to reproduce", tried)
	}
	want := found.Digest()
	t.Logf("reproducing seed %s (trace %s) %d times", found.Seed, found.TraceHash, replays)

	for i := 0; i < replays; i++ {
		got := Run(found.Seed, opt)
		if got.SimErr != nil {
			t.Fatalf("replay %d of seed %s: simulation failed: %v", i+1, found.Seed, got.SimErr)
		}
		if !got.Violated() {
			t.Fatalf("replay %d of seed %s: the violation did not reproduce (status %v).\n"+
				"A seed that does not replay is not a reproduction recipe, and every failure "+
				"report lockstep prints is worthless if this can happen.",
				i+1, found.Seed, got.Check.Status)
		}
		if got.TraceHash != found.TraceHash {
			t.Fatalf("replay %d of seed %s: trace hash %s, want %s.\n"+
				"The same seed explored a different schedule: something in this example is "+
				"drawing from a source other than Sim.Rand, iterating a map, or reading the "+
				"host clock.",
				i+1, found.Seed, got.TraceHash, found.TraceHash)
		}
		if diff := got.Digest(); diff != want {
			t.Fatalf("replay %d of seed %s: the failure report changed.\n\n--- first run ---\n%s\n--- replay ---\n%s",
				i+1, found.Seed, want, diff)
		}
	}
}

// TestFaultsAreActuallyInjected checks the thing every chaos test eventually
// stops doing.
//
// Fault injection rots silently: a probability gets scaled to nothing, a
// partition is healed a line too early, a crash is aimed at a node that has
// already finished. The suite stays green because the happy path still works,
// and the tests go on measuring nothing. This asserts every fault this example
// claims to inject actually reached the run.
func TestFaultsAreActuallyInjected(t *testing.T) {
	out := Run(firstSeed, DefaultOptions(QuorumReads))
	if out.SimErr != nil {
		t.Fatalf("seed %s: simulation failed: %v", out.Seed, out.SimErr)
	}
	assertFaultsFired(t, out)

	if !out.Net.Balanced() {
		t.Errorf("network accounting does not balance: %s", out.Net)
	}
	if out.VirtualTime <= 0 {
		t.Errorf("no virtual time elapsed; the run cannot have done anything")
	}
	t.Logf("seed %s: %s; faults: %s", out.Seed, out.Net, strings.Join(out.Faults, "; "))
}

// assertFaultsFired fails unless the run really did lose messages, duplicate
// messages, delay them, isolate a replica and kill one.
func assertFaultsFired(t *testing.T, out Outcome) {
	t.Helper()
	if out.Net.Dropped == 0 {
		t.Errorf("seed %s: no message was dropped; loss injection is not reaching the run (%s)",
			out.Seed, out.Net)
	}
	if out.Net.Duplicated == 0 {
		t.Errorf("seed %s: no message was duplicated (%s)", out.Seed, out.Net)
	}
	if out.Net.DelayedTotal == 0 {
		t.Errorf("seed %s: no latency was injected (%s)", out.Seed, out.Net)
	}
	log := strings.Join(out.Faults, "; ")
	for _, want := range []string{"isolated", "healed", "crashed", "restarted"} {
		if !strings.Contains(log, want) {
			t.Errorf("seed %s: fault log does not mention %q: %s", out.Seed, want, log)
		}
	}
}

// TestReplicasDivergeUnderFaults establishes the precondition the bug needs,
// separately from the bug itself.
//
// Single-replica reads are only unsafe if replicas can disagree, so if this
// example ever stopped producing divergence — a timeout raised, a fault rate
// lowered — the headline test would go green for a reason that has nothing to do
// with the read path being fixed. This checks the mechanism directly, in both
// modes, so that neither can go quiet unnoticed.
func TestReplicasDivergeUnderFaults(t *testing.T) {
	for _, mode := range []Mode{SingleReplicaReads, QuorumReads} {
		diverged := false
		Sweep(firstSeed, 16, DefaultOptions(mode), func(out Outcome) bool {
			diverged = out.Diverged
			return !diverged
		})
		if !diverged {
			t.Errorf("%s: no two replicas ever held different values for the same key across "+
				"16 seeds; the workload can no longer produce the state that makes "+
				"single-replica reads unsafe", mode)
		}
	}
}
