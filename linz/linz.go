// Package linz decides whether a recording of concurrent operations could have
// come from a correct implementation — that is, whether it is linearizable.
//
// It is the other half of lockstep's argument. The sim package makes a
// distributed system's schedule reproducible; this package decides whether what
// happened under that schedule was allowed. Neither is much use alone: a
// reproducible run you cannot judge is a story with no ending, and a violation
// you cannot reproduce is a rumour.
//
// linz does not import sim and never will. A history is just data — operations
// with timestamps — so this checker works equally well on a recording from a
// deterministic simulation, from a stress test running against a real cluster,
// or from a server-side log parsed after the fact.
//
// # What linearizability actually means
//
// A concurrent operation is not an instant. It has an invocation — "client 3
// called write(7) at t=10" — and a response — "and was told it succeeded at
// t=25" — and in between, the client knows nothing about when, or even whether,
// the operation took effect. If two clients' operations overlap in time, either
// might have gone first.
//
// A history is linearizable if you can choose one instant inside each
// operation's interval — its linearization point — such that running the
// operations one at a time in that order, against an ordinary single-threaded
// implementation of the same data type, produces exactly the outputs that were
// actually observed.
//
// Informally: the system behaved as if it were a single object handling one
// request at a time, and each request took effect at some moment while it was in
// flight. That is the strongest single-object consistency condition there is, and
// it is the one worth testing for, because it is the only one under which a
// programmer can reason about a distributed store the way they reason about a
// variable.
//
// Two consequences deserve to be stated separately.
//
// Real time is binding. If A's response happened before B's invocation, then A
// must come before B in the ordering. Overlapping operations may be reordered
// freely; non-overlapping ones may not. Drop that requirement and you have
// sequential consistency, under which a value written and acknowledged an hour
// ago may still legally be invisible to everyone. Real-time ordering is what
// makes linearizability worth anything, and it is the entire reason a history
// must record two timestamps per operation rather than one.
//
// Linearizability is a property of a whole history, never of one operation.
// There is no such thing as "this read returned the wrong value". A read is
// wrong only relative to a placement of every other operation in the history —
// which is why checking is a search problem and not a scan.
//
// # Why the obvious checks are wrong
//
// The tempting approach is: replay the history in order and check each response
// as you go. Every variant of it is wrong, and each is wrong in an instructive
// way.
//
// Sorting by invocation time and replaying rejects correct systems. Two
// overlapping writes may take effect in either order; if you fix one order and
// the observed reads imply the other, you report a violation that is not there.
// Sorting by response time has the same flaw. There is no ordering of the
// endpoints that is correct in general, because for overlapping operations the
// history genuinely does not say which came first.
//
// Being greedy — at each step, linearise any operation that is legal right now —
// also rejects correct systems, and this is the subtle one. A choice that is
// legal in isolation can make the rest of the history impossible, while a
// different legal choice would have worked. Concretely, with a register starting
// at 0:
//
//	client 0: write(1)   invoked t=0,  returned t=100
//	client 1: write(2)   invoked t=0,  returned t=100
//	client 2: read() -> 2  invoked t=10, returned t=20
//	client 3: read() -> 1  invoked t=30, returned t=40
//
// A greedy checker starts by placing write(1), because writes are always legal.
// It then cannot explain the read of 2 followed by the read of 1, and reports a
// violation. But the history is fine: write(2), read 2, write(1), read 1 — both
// writes were still in flight, so both could take effect exactly where they are
// needed. Finding that requires undoing the first choice. Backtracking is not an
// optimisation here; it is the difference between a correct checker and a
// broken one.
//
// Finally, deciding an operation "must" have taken effect at its response is
// wrong for operations that never returned at all — and those are exactly the
// operations a crash test produces.
//
// # Operations that never returned
//
// When a client crashes mid-call, the write may have been durably applied before
// the crash, or may have been dropped, and no one will ever know which. A checker
// that assumes it happened will report violations that are not real; one that
// assumes it did not will miss violations that are. The definition (Herlihy and
// Wing, 1990) says to try both: a history is linearizable if you can append
// responses to SOME subset of the still-pending invocations, discard the rest,
// and linearise what remains.
//
// This package implements exactly that. A pending operation is given a response
// time later than every real timestamp, so it may be placed anywhere at or after
// its invocation, and its output is reported to the model as NoResponse, meaning
// "no observed value can contradict you". Discarding is handled too: the moment
// every operation that did return has been placed, the search stops and declares
// success, because whatever is left was never observed by anyone and is free to
// have had no effect. Both possibilities, without enumerating subsets.
//
// # How the checker works
//
// The algorithm is Wing and Gong's, with the memoisation that makes it practical
// (the same shape as Porcupine, and described in Lowe's "Testing for
// Linearizability", 2017).
//
// Every invocation and response goes on a doubly linked list in time order. The
// search walks from the head, trying to linearise the first operation it can,
// removing that operation's two entries and restarting; when it reaches a
// response whose operation is still unplaced it has hit a real-time barrier and
// must backtrack. The list encodes the real-time constraint structurally, so
// there is no separate check for it anywhere in this package.
//
// On its own that is a factorial search. What makes it usable is memoising
// visited (set of already-placed operations, resulting state) pairs: two paths
// that have placed the same operations and reached the same state face an
// identical remaining problem, so the second one stops immediately. The tree of
// orderings collapses into a graph of reachable states.
//
// The result is cost that is roughly linear in the number of operations and
// exponential in how many of them overlap in time. That second term is the one
// that bites, and it is why partitioning matters so much.
//
// # Partitioning, or why this is fast enough to use
//
// Linearizability is compositional: a history over several independent objects
// is linearizable exactly when each object's sub-history is linearizable on its
// own. So a workload over 100 keys is 100 small problems, not one enormous one.
// A model that implements Partitioner gets this automatically.
//
// Measured on the same key-value histories checked both ways (BenchmarkPartitioned):
//
//	operations   partitioned   as one problem
//	       200        254 µs           955 µs
//	       500        482 µs           359 ms
//	     1,000        870 µs           505 ms
//	     2,000        1.8 ms            90 ms
//
// Two things in that table matter more than the headline ratio. The partitioned
// column is linear — twice the operations, twice the time — because splitting by
// key keeps each sub-problem the same small size however long the run gets. The
// other column is not even monotone: 500 operations cost four times what 2,000
// did, because the unpartitioned search's cost depends on how ambiguous the
// interleaving happens to be rather than on its length. Unpredictable is worse
// than slow; a check that usually takes a second and occasionally takes an hour
// cannot be put in CI.
//
// Compositionality is also why the partitioned check is not merely faster but
// better: a violation found on key "b" is reported against the three operations
// that touch "b", not against the two thousand that do not.
//
// # What is affordable, measured
//
// All figures from the benchmarks in this package, on an Intel Xeon at 2.10 GHz
// (linux/amd64, Go 1.24.7). The checker is single-threaded, so these are
// single-core numbers.
//
// Cost against history length, which is the easy axis:
//
//	100,000 operations, 1 client (no overlap at all):    53 ms
//	 10,000 operations, 1 client:                       7.4 ms
//	  6,400 operations, 4 clients:                       13 ms
//	  1,600 operations, 4 clients:                      1.8 ms
//
// Cost against concurrency width, which is the hard axis. One hundred operations
// on ONE register, every one of them linearizable, as the number of clients
// hammering it rises:
//
//	clients      time     model steps
//	      2     43 µs             123
//	      4     77 µs             485
//	      6    394 µs           3,838
//	      8    1.3 ms          14,453
//	     10    5.5 ms          62,914
//	     12     43 ms         442,662
//	     14    189 ms       1,844,298
//	     16    356 ms       3,256,553
//	     18    7.4 s       44,840,949
//	     20     33 s      178,003,060
//
// Every two extra clients multiplies the work by roughly four. That is the
// exponential, measured. Refuting a broken history at the same width costs about
// five times more again — 219 ms at 12 clients against 43 ms — because finding
// one valid ordering can get lucky and proving that none exists cannot.
//
// So the practical wall is concurrency per partition, and for a hundred
// operations it stands at about 18 clients on one key: that check takes seven
// seconds and 45 million model steps, which is already most of DefaultMaxSteps.
// At 20 clients the default budget is exhausted and the honest answer becomes
// Unknown. No amount of tuning moves that wall far — the problem is NP-hard in
// general (Gibbons and Korach, 1997) — and this package would rather tell you it
// gave up than pretend.
//
// Length is nearly free by comparison: 100,000 operations at width 1 costs 53 ms
// and 6,400 at width 4 costs 13 ms. So the way to stay inside the affordable
// region is to partition by key, keep the clients touching any single key in the
// single digits, and make histories LONGER rather than WIDER. Ten thousand
// operations at width 4 are checked in milliseconds and will find more bugs than
// two hundred at width 30 that never finish.
//
// # Using it
//
//	h := linz.NewHistory()
//	// ... from several goroutines, under -race:
//	h.Do(clientID, linz.Put("x", "1"), func() any { store.Put("x", "1"); return nil })
//	h.Do(clientID, linz.Get("x"), func() any { return store.Get("x") })
//	// ...
//	if res := linz.Check(linz.NewKVModel(), h); !res.OK() {
//	    t.Fatal(res)
//	}
//
// Check applies a default budget, so a pathological history fails loudly instead
// of hanging CI. Read the Unknown documentation before you decide how to treat
// that verdict.
package linz

import (
	"fmt"
	"reflect"
	"sort"
	"time"
)

// Budget bounds the search so that a history the checker cannot decide fails
// loudly instead of hanging.
//
// The zero value is not "unlimited" — it means "use the defaults". That choice
// is deliberate and matches the rest of lockstep: a configuration field whose
// zero value is the dangerous setting eventually gets left out of a struct
// literal by someone in a hurry, and the failure is a CI job that runs until the
// timeout with no explanation. Ask for unlimited explicitly, with a negative
// value, and only when you mean it.
type Budget struct {
	// MaxSteps caps the number of model transitions the search may evaluate.
	// Zero means DefaultMaxSteps; negative means unlimited.
	//
	// Prefer this over Timeout. It is a property of the history and the model,
	// so the same check on the same input reaches the same verdict on a laptop
	// and on a loaded CI runner. A wall-clock limit does not have that property,
	// and a test that passes or fails depending on what else the machine is doing
	// is a test people learn to ignore.
	MaxSteps int64

	// Timeout caps wall-clock time. Zero means DefaultTimeout; negative means
	// unlimited.
	//
	// It is a backstop, not the primary control: the point is that a model with
	// an unexpectedly expensive Step cannot blow through a step budget slowly
	// enough to still hang the build.
	Timeout time.Duration
}

// DefaultMaxSteps is the step budget Check uses.
//
// Chosen from measurement rather than taste. The search sustains between five
// and ten million model transitions per second on the benchmark machine with the
// built-in models — the lower end once the memo table has grown large enough to
// miss cache — so this is five to ten seconds of work. That is long enough to
// decide every history in this package's benchmarks up to 18 concurrent clients
// on one key, and short enough that a pathological one comes back Unknown while
// someone is still watching the build.
const DefaultMaxSteps int64 = 50_000_000

// DefaultTimeout is the wall-clock backstop Check uses.
const DefaultTimeout = 30 * time.Second

func (b Budget) withDefaults() Budget {
	if b.MaxSteps == 0 {
		b.MaxSteps = DefaultMaxSteps
	}
	if b.Timeout == 0 {
		b.Timeout = DefaultTimeout
	}
	return b
}

// clockCheckInterval is how many search steps pass between wall-clock checks.
//
// A time.Now on every step would cost a large fraction of the step itself — the
// search runs five to ten million of them per second — and would make the
// checker's own throughput depend on the platform's clock implementation.
// Checking every few thousand steps costs nothing measurable, and at that rate
// overshoots the deadline by well under a millisecond.
const clockCheckInterval = 4096

// budgetState tracks consumption across every partition of one check.
type budgetState struct {
	maxSteps  int64
	steps     int64
	deadline  time.Time
	timeout   time.Duration
	timed     bool
	countdown int64
	reason    string
}

func newBudgetState(b Budget, start time.Time) *budgetState {
	b = b.withDefaults()
	s := &budgetState{maxSteps: b.MaxSteps, countdown: clockCheckInterval}
	if b.Timeout > 0 {
		s.deadline = start.Add(b.Timeout)
		s.timeout = b.Timeout
		s.timed = true
	}
	return s
}

// tick charges one model transition to the budget and reports whether it is
// spent.
//
// The limit is tested before the counter moves, so Result.Steps never exceeds
// the budget it was given. A report that says "stopped after 3 steps (budget:
// 2)" invites a reader to go looking for an off-by-one in the checker, and that
// reader's time is worth more than the branch this costs.
func (b *budgetState) tick() bool {
	if b.maxSteps > 0 && b.steps >= b.maxSteps {
		b.reason = fmt.Sprintf("search stopped after %d model steps (budget: %d)", b.steps, b.maxSteps)
		return true
	}
	b.steps++
	b.countdown--
	if b.countdown <= 0 {
		b.countdown = clockCheckInterval
		if b.timed && time.Now().After(b.deadline) {
			b.reason = fmt.Sprintf("search stopped after %d model steps: wall-clock budget of %s exhausted",
				b.steps, b.timeout)
			return true
		}
	}
	return false
}

// Check decides whether a history is linearizable with respect to a model,
// using DefaultMaxSteps and DefaultTimeout.
//
// It is the function to call. Use CheckWithBudget when the defaults are wrong
// for your workload — and note that "the check comes back Unknown" is usually a
// message about the workload, not about the budget.
func Check(m Model, h *History) Result {
	return CheckWithBudget(m, h, Budget{})
}

// CheckWithBudget is Check with an explicit bound on how much work the search
// may do.
//
// The budget covers the whole check, not each partition, because what a caller
// wants to bound is how long this call can take. Partitions are searched in a
// deterministic order (sorted by key) so that a budget exhausted halfway through
// stops at the same place every time — an Unknown verdict that moved around
// between runs would be impossible to act on.
//
// The search stops at the first violation it finds. One violation, explained
// well, is what fixes a bug; five, explained badly, is what gets a checker
// switched off.
func CheckWithBudget(m Model, h *History, b Budget) Result {
	start := time.Now()
	ops := h.Operations()
	budget := newBudgetState(b, start)

	groups, partitioned := partitionOps(m, ops)
	describer, hasDescriber := extension[Describer](m)

	res := Result{
		Status:     Linearizable,
		Operations: len(ops),
		Partitions: len(groups),
	}
	for _, op := range ops {
		if op.Pending {
			res.Pending++
		}
	}

	// Every searcher is built before any of them runs. Building one lays its
	// partition out on a timeline, which is where the concurrency width comes
	// from, and the alternative — measuring the width separately — meant a second
	// sort of every partition on a path where sorting is already most of the cost
	// of a short check. Doing it up front also means the reported width covers
	// the whole history rather than only the partitions checked before the first
	// violation.
	searchers := make([]*searcher, len(groups))
	for i, g := range groups {
		searchers[i] = newSearcher(m, g.ops, budget)
		if len(g.ops) > res.Largest {
			res.Largest = len(g.ops)
		}
		if searchers[i].width > res.Widest {
			res.Widest = searchers[i].width
		}
	}

	for i, s := range searchers {
		switch s.run() {
		case searchOK:
			continue
		case searchBudget:
			res.Status = Unknown
			res.Reason = budget.reason
		case searchFailed:
			res.Status = NotLinearizable
			res.Failure = buildFailure(s, groups[i], partitioned, describer, hasDescriber)
		}
		break
	}

	res.Steps = budget.steps
	res.Elapsed = time.Since(start)
	return res
}

// group is one partition: the operations touching a single independent object.
type group struct {
	key    any
	hasKey bool
	ops    []Operation
}

// partitionOps splits a history into independent sub-problems, if the model says
// how.
//
// The sort at the end is not cosmetic. Map iteration order in Go is deliberately
// randomised, so without it the partition a failing check reports would change
// from run to run, and two runs of the same failing test would print different
// stories about the same bug.
func partitionOps(m Model, ops []Operation) ([]group, bool) {
	p, ok := extension[Partitioner](m)
	if !ok {
		return []group{{ops: ops}}, false
	}

	index := make(map[any]int)
	var groups []group
	for _, op := range ops {
		key := p.PartitionKey(op.Input)
		requireComparable(key)
		i, seen := index[key]
		if !seen {
			i = len(groups)
			index[key] = i
			groups = append(groups, group{key: key, hasKey: true})
		}
		groups[i].ops = append(groups[i].ops, op)
	}
	sort.SliceStable(groups, func(i, j int) bool {
		return fmt.Sprint(groups[i].key) < fmt.Sprint(groups[j].key)
	})
	if groups == nil {
		groups = []group{{ops: nil}}
	}
	return groups, true
}

// requireComparable turns an obscure runtime panic deep inside a map assignment
// into a message that names the mistake.
func requireComparable(key any) {
	if key == nil {
		return
	}
	if !reflect.TypeOf(key).Comparable() {
		panic(fmt.Sprintf("linz: PartitionKey returned %T, which is not comparable and "+
			"cannot be used as a partition key; return a string, an int, or a struct of those", key))
	}
}

// buildFailure turns the search's record of its deepest dead end into a report.
func buildFailure(s *searcher, g group, partitioned bool, d Describer, hasDescriber bool) *Failure {
	// Operations are read back out of the searcher rather than the group,
	// because the searcher renumbers them into invocation order and its recorded
	// ids refer to that numbering.
	f := &Failure{
		PartitionKey: g.key,
		Partitioned:  partitioned && g.hasKey,
		State:        s.deadState,
		Total:        len(s.ops),
	}
	if hasDescriber {
		f.describe = d.DescribeOperation
		f.describeState = d.DescribeState
	}
	if !s.haveDeadEnd {
		// The search failed without ever reaching a dead end, which can only
		// happen for an empty partition. Nothing useful to say.
		return f
	}
	for _, id := range s.deadOrder {
		f.Prefix = append(f.Prefix, s.ops[id])
	}
	for _, b := range s.deadBlocked {
		f.Blocked = append(f.Blocked, s.ops[b.id])
		f.BlockedRejected = append(f.BlockedRejected, b.rejected)
	}
	if s.deadBarrier >= 0 {
		op := s.ops[s.deadBarrier]
		f.Barrier = &op
	}
	return f
}
