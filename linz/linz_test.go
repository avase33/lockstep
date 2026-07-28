package linz

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
	"testing"
	"time"
)

// unlimited is the budget benchmarks and "how much work is this really" tests
// use. Ordinary tests go through Check so that the default budget is exercised
// too.
var unlimited = Budget{MaxSteps: -1, Timeout: -1}

func mustCheck(t *testing.T, m Model, ops []Operation, want Status) Result {
	t.Helper()
	res := Check(m, FromOperations(ops))
	if res.Status != want {
		t.Fatalf("got %s, want %s\n%s", res.Status, want, res)
	}
	return res
}

// ---------------------------------------------------------------------------
// The easy cases, which are worth having because a checker that cannot do these
// is not worth debugging on the hard ones.
// ---------------------------------------------------------------------------

func TestSequentialHistoryIsLinearizable(t *testing.T) {
	// No two operations overlap, so there is exactly one candidate ordering and
	// it is the one the outputs were generated from.
	ops := []Operation{
		{ClientID: 0, Input: Read(), Output: 0, Call: 0, Return: 10},
		{ClientID: 0, Input: Write(7), Output: nil, Call: 20, Return: 30},
		{ClientID: 1, Input: Read(), Output: 7, Call: 40, Return: 50},
		{ClientID: 1, Input: CAS(7, 9), Output: true, Call: 60, Return: 70},
		{ClientID: 0, Input: Read(), Output: 9, Call: 80, Return: 90},
		{ClientID: 1, Input: CAS(7, 11), Output: false, Call: 100, Return: 110},
		{ClientID: 0, Input: Read(), Output: 9, Call: 120, Return: 130},
	}
	res := mustCheck(t, NewRegisterModel(), ops, Linearizable)
	if res.Widest != 1 {
		t.Fatalf("a sequential history should have max concurrency 1, got %d", res.Widest)
	}
}

func TestEmptyHistoryIsLinearizable(t *testing.T) {
	mustCheck(t, NewRegisterModel(), nil, Linearizable)
	mustCheck(t, NewKVModel(), nil, Linearizable)
}

// A write completes, and a later read — invoked strictly after that write
// returned — does not see it. No ordering can explain that, because real time
// forbids moving the read before the write.
func TestStaleReadAfterCompletedWriteIsNotLinearizable(t *testing.T) {
	ops := []Operation{
		{ClientID: 0, Input: Write(1), Output: nil, Call: 0, Return: 10},
		{ClientID: 1, Input: Read(), Output: 0, Call: 20, Return: 30},
	}
	res := mustCheck(t, NewRegisterModel(), ops, NotLinearizable)

	// The report has to be specific enough to act on, which is the whole point
	// of building one.
	report := res.Failure.String()
	for _, want := range []string{"write(1)", "read() -> 0", "state after those operations: 1"} {
		if !strings.Contains(report, want) {
			t.Errorf("failure report does not mention %q:\n%s", want, report)
		}
	}
	if len(res.Failure.Prefix) != 1 {
		t.Errorf("longest linearizable prefix should be the write alone, got %d operations", len(res.Failure.Prefix))
	}
	if len(res.Failure.Blocked) != 1 || res.Failure.Blocked[0].Input.(RegisterInput).Op != OpRead {
		t.Errorf("the read should be the operation that could not be placed, got %v", res.Failure.Blocked)
	}
	if !res.Failure.BlockedRejected[0] {
		t.Errorf("the read should be reported as rejected by the model, not pruned by the cache")
	}
}

// A read returns a value that is only ever written by an operation invoked long
// afterwards. Nothing can be reordered across that gap.
func TestReadOfAValueWrittenLaterIsNotLinearizable(t *testing.T) {
	ops := []Operation{
		{ClientID: 0, Input: Read(), Output: 5, Call: 10, Return: 20},
		{ClientID: 1, Input: Write(5), Output: nil, Call: 100, Return: 110},
	}
	res := mustCheck(t, NewRegisterModel(), ops, NotLinearizable)
	if len(res.Failure.Prefix) != 0 {
		t.Fatalf("no operation should have been placeable first, got prefix %v", res.Failure.Prefix)
	}
}

// ---------------------------------------------------------------------------
// The case that separates a real checker from a plausible-looking one.
// ---------------------------------------------------------------------------

// Two writes overlap everything. Among the orderings the checker may consider,
// exactly one explains both reads, and the greedy choice is not it: taking
// write(1) first (it is legal, and it comes first in the history) leads to a
// dead end. Only backtracking finds write(2), read 2, write(1), read 1.
func TestConcurrentWritesRequireBacktracking(t *testing.T) {
	ops := []Operation{
		{ClientID: 0, Input: Write(1), Output: nil, Call: 0, Return: 100},
		{ClientID: 1, Input: Write(2), Output: nil, Call: 0, Return: 100},
		{ClientID: 2, Input: Read(), Output: 2, Call: 10, Return: 20},
		{ClientID: 3, Input: Read(), Output: 1, Call: 30, Return: 40},
	}
	res := mustCheck(t, NewRegisterModel(), ops, Linearizable)
	// Both writes plus whichever read is in flight: three at a time. The reads
	// never overlap each other, which is exactly what pins their order and makes
	// the history hard.
	if res.Widest != 3 {
		t.Fatalf("expected max concurrency 3, got %d", res.Widest)
	}
}

// The same shape, one step harder: three overlapping writes and three reads
// that pin them into an order which is the exact reverse of the order the
// writes appear in the history.
func TestManyInvalidOrderingsOneValid(t *testing.T) {
	ops := []Operation{
		{ClientID: 0, Input: Write(1), Output: nil, Call: 0, Return: 1000},
		{ClientID: 1, Input: Write(2), Output: nil, Call: 0, Return: 1000},
		{ClientID: 2, Input: Write(3), Output: nil, Call: 0, Return: 1000},
		{ClientID: 3, Input: Read(), Output: 3, Call: 10, Return: 20},
		{ClientID: 4, Input: Read(), Output: 2, Call: 30, Return: 40},
		{ClientID: 5, Input: Read(), Output: 1, Call: 50, Return: 60},
	}
	mustCheck(t, NewRegisterModel(), ops, Linearizable)

	// Change one read to a value nobody ever wrote and the same history becomes
	// impossible, which is what makes the test above meaningful.
	bad := append([]Operation(nil), ops...)
	bad[5] = Operation{ClientID: 5, Input: Read(), Output: 4, Call: 50, Return: 60}
	mustCheck(t, NewRegisterModel(), bad, NotLinearizable)
}

// ---------------------------------------------------------------------------
// Operations that never returned.
// ---------------------------------------------------------------------------

// A client crashed while writing, and a later read saw the value. The history
// only makes sense if the crashed write did take effect, so the checker must be
// willing to place an operation that has no response.
func TestPendingOperationTreatedAsHavingTakenEffect(t *testing.T) {
	ops := []Operation{
		{ClientID: 0, Input: Write(1), Call: 0, Pending: true},
		{ClientID: 1, Input: Read(), Output: 1, Call: 10, Return: 20},
	}
	mustCheck(t, NewRegisterModel(), ops, Linearizable)

	// And it must not be forced to place it: here the same crashed write must
	// end up ordered after the read (equivalently, may be treated as never
	// having happened), or the read of 0 is unexplainable.
	ops = []Operation{
		{ClientID: 0, Input: Write(1), Call: 0, Pending: true},
		{ClientID: 1, Input: Read(), Output: 0, Call: 10, Return: 20},
	}
	mustCheck(t, NewRegisterModel(), ops, Linearizable)
}

// mutexModel is a specification with a PARTIAL operation: locking a held mutex
// is not "an operation that returns false", it is something a correct
// implementation can never do. That is what makes it the right model for
// testing the other half of pending-operation handling — a history that is only
// explicable if the crashed operation is discarded entirely, because there is
// nowhere in the ordering it could legally go.
//
// It implements Model directly rather than Deterministic, which also exercises
// the raw interface. Note that it never looks at the output, so NoResponse needs
// no special case here; a model whose outputs carry information would need one.
type mutexModel struct{}

func (mutexModel) Init() State { return false }

func (mutexModel) Equal(a, b State) bool { return a.(bool) == b.(bool) }

func (mutexModel) Step(state State, input, _ any) (bool, State) {
	held := state.(bool)
	switch input.(string) {
	case "lock":
		if held {
			return false, nil
		}
		return true, true
	case "unlock":
		if !held {
			return false, nil
		}
		return true, false
	}
	panic("unknown mutex operation")
}

func (mutexModel) DescribeOperation(input, _ any) string { return input.(string) + "()" }

func (mutexModel) DescribeState(s State) string {
	if s.(bool) {
		return "held"
	}
	return "free"
}

func TestPendingOperationTreatedAsNeverHavingHappened(t *testing.T) {
	// Client 1 acquires the mutex and holds it. Client 0 crashed inside its own
	// acquire. There is no legal position for client 0's lock: before client 1's
	// it makes that one illegal, after it the mutex is already held. The history
	// is linearizable only because a pending operation may be discarded.
	ops := []Operation{
		{ClientID: 0, Input: "lock", Call: 0, Pending: true},
		{ClientID: 1, Input: "lock", Output: nil, Call: 10, Return: 20},
	}
	mustCheck(t, mutexModel{}, ops, Linearizable)

	// The same history with both operations completed is impossible, which
	// proves the model really does reject double locking and that the case above
	// passed for the right reason.
	ops[0] = Operation{ClientID: 0, Input: "lock", Output: nil, Call: 0, Return: 30}
	mustCheck(t, mutexModel{}, ops, NotLinearizable)
}

func TestPendingOperationsAreCounted(t *testing.T) {
	ops := []Operation{
		{ClientID: 0, Input: Write(1), Call: 0, Pending: true},
		{ClientID: 1, Input: Read(), Output: 1, Call: 10, Return: 20},
		{ClientID: 2, Input: Read(), Call: 15, Pending: true},
	}
	res := mustCheck(t, NewRegisterModel(), ops, Linearizable)
	if res.Pending != 2 {
		t.Fatalf("Pending = %d, want 2", res.Pending)
	}
}

// ---------------------------------------------------------------------------
// Partitioning.
// ---------------------------------------------------------------------------

func TestPartitioningGivesTheSameVerdict(t *testing.T) {
	rng := rand.New(rand.NewSource(20240728))

	// A range of shapes, so this is not one lucky history: some clean, some with
	// a corrupted response, over several keys and concurrency widths.
	for _, keys := range []int{1, 3, 17} {
		for _, clients := range []int{1, 4, 7} {
			for _, corrupt := range []bool{false, true} {
				ops := randomKVHistory(rng, 60, keys, clients)
				want := Linearizable
				if corrupt {
					corruptOneGet(t, rng, ops)
					want = NotLinearizable
				}
				h := FromOperations(ops)

				split := CheckWithBudget(NewKVModel(), h, unlimited)
				whole := CheckWithBudget(Unpartitioned(NewKVModel()), h, unlimited)

				if split.Status != want || whole.Status != want {
					t.Fatalf("keys=%d clients=%d corrupt=%v: partitioned %s, unpartitioned %s, want %s\n%s",
						keys, clients, corrupt, split.Status, whole.Status, want, split)
				}
				if whole.Partitions != 1 {
					t.Fatalf("Unpartitioned should produce a single partition, got %d", whole.Partitions)
				}
				if keys > 1 && split.Partitions < 2 {
					t.Fatalf("expected the key-value model to split %d keys, got %d partitions", keys, split.Partitions)
				}
			}
		}
	}
}

func TestPartitionCountAndWidthAreReported(t *testing.T) {
	ops := randomKVHistory(rand.New(rand.NewSource(1)), 200, 10, 6)
	res := CheckWithBudget(NewKVModel(), FromOperations(ops), unlimited)
	if !res.OK() {
		t.Fatalf("generated history should be linearizable:\n%s", res)
	}
	if res.Partitions != 10 {
		t.Fatalf("Partitions = %d, want 10", res.Partitions)
	}
	if res.Largest >= res.Operations {
		t.Fatalf("Largest partition (%d) should be smaller than the whole history (%d)", res.Largest, res.Operations)
	}
	if res.Widest < 1 || res.Widest > 6 {
		t.Fatalf("Widest = %d, want between 1 and the client count", res.Widest)
	}
}

// ---------------------------------------------------------------------------
// Budget.
// ---------------------------------------------------------------------------

// The assertion that matters most in this file: running out of budget must not
// look like success. A checker that reports "linearizable" when it means "I gave
// up" turns every history hard enough to be interesting into a green test.
func TestBudgetExhaustionIsUnknownNotLinearizable(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	ops := randomRegisterHistory(rng, 80, 10)

	full := CheckWithBudget(NewRegisterModel(), FromOperations(ops), unlimited)
	if full.Status != Linearizable {
		t.Fatalf("generated history should be linearizable, got %s\n%s", full.Status, full)
	}
	if full.Steps < 100 {
		t.Fatalf("history is too easy to be a useful budget test: %d steps", full.Steps)
	}

	// Half the work it actually needs, so exhaustion is certain.
	starved := CheckWithBudget(NewRegisterModel(), FromOperations(ops), Budget{MaxSteps: full.Steps / 2})
	if starved.Status != Unknown {
		t.Fatalf("starved check returned %s, want unknown (it must never claim success)\n%s", starved.Status, starved)
	}
	if starved.OK() {
		t.Fatal("Result.OK must be false for an undecided check")
	}
	if starved.Reason == "" {
		t.Fatal("an unknown verdict must explain itself")
	}
	if !strings.Contains(starved.String(), "unknown") {
		t.Fatalf("rendered result should say unknown:\n%s", starved)
	}

	// The same applies to a history that is genuinely broken: without the budget
	// to prove it, the honest answer is still "unknown".
	bad := append([]Operation(nil), ops...)
	bad[len(bad)-1].Output = 1 << 30
	if starvedBad := CheckWithBudget(NewRegisterModel(), FromOperations(bad), Budget{MaxSteps: 20}); starvedBad.Status != Unknown {
		t.Fatalf("starved check of a broken history returned %s, want unknown", starvedBad.Status)
	}
}

func TestTimeoutIsReportedAsUnknown(t *testing.T) {
	// Wide enough that refuting it takes far longer than the timeout allows.
	ops := randomRegisterHistory(rand.New(rand.NewSource(7)), 400, 26)
	ops[len(ops)-1].Output = 1 << 30
	res := CheckWithBudget(NewRegisterModel(), FromOperations(ops), Budget{MaxSteps: -1, Timeout: 20 * time.Millisecond})
	if res.Status != Unknown {
		t.Fatalf("got %s, want unknown", res.Status)
	}
	if res.Elapsed > 2*time.Second {
		t.Fatalf("timeout overshot badly: %s", res.Elapsed)
	}
}

func TestZeroBudgetMeansDefaultsNotUnlimited(t *testing.T) {
	b := Budget{}.withDefaults()
	if b.MaxSteps != DefaultMaxSteps || b.Timeout != DefaultTimeout {
		t.Fatalf("zero Budget must mean defaults, got %+v", b)
	}
	neg := Budget{MaxSteps: -1, Timeout: -1}.withDefaults()
	if neg.MaxSteps != -1 || neg.Timeout != -1 {
		t.Fatalf("negative Budget must stay unlimited, got %+v", neg)
	}
}

// ---------------------------------------------------------------------------
// Property test: generate histories that are linearizable by construction,
// then break exactly one thing.
// ---------------------------------------------------------------------------

func TestPropertyGeneratedHistoriesAreAccepted(t *testing.T) {
	for seed := int64(0); seed < 300; seed++ {
		rng := rand.New(rand.NewSource(seed))
		clients := 1 + rng.Intn(6)
		keys := 1 + rng.Intn(5)
		ops := randomKVHistory(rng, 40+rng.Intn(60), keys, clients)

		res := CheckWithBudget(NewKVModel(), FromOperations(ops), unlimited)
		if res.Status != Linearizable {
			t.Fatalf("seed %d (%d clients, %d keys): interleaving a valid sequential history "+
				"produced a verdict of %s\n%s\n%s", seed, clients, keys, res.Status, res, FromOperations(ops))
		}
	}
}

func TestPropertyCorruptedHistoriesAreRejected(t *testing.T) {
	for seed := int64(0); seed < 300; seed++ {
		rng := rand.New(rand.NewSource(seed))
		clients := 1 + rng.Intn(6)
		keys := 1 + rng.Intn(5)
		ops := randomKVHistory(rng, 40+rng.Intn(60), keys, clients)
		corruptOneGet(t, rng, ops)

		res := CheckWithBudget(NewKVModel(), FromOperations(ops), unlimited)
		if res.Status != NotLinearizable {
			t.Fatalf("seed %d: a read returning a value nobody ever wrote was accepted (%s)\n%s",
				seed, res.Status, res)
		}
	}
}

// The register model has no partitioner, so this covers the other search shape:
// one big problem rather than many small ones.
func TestPropertyRegisterHistories(t *testing.T) {
	for seed := int64(0); seed < 120; seed++ {
		rng := rand.New(rand.NewSource(seed))
		clients := 1 + rng.Intn(5)
		ops := randomRegisterHistory(rng, 30+rng.Intn(40), clients)

		if res := CheckWithBudget(NewRegisterModel(), FromOperations(ops), unlimited); res.Status != Linearizable {
			t.Fatalf("seed %d: %s\n%s", seed, res.Status, res)
		}

		// Corrupt one read to a value that is never written anywhere in the
		// history, so no ordering whatsoever can produce it.
		idx := -1
		for i := range ops {
			if in := ops[i].Input.(RegisterInput); in.Op == OpRead {
				idx = i
			}
		}
		if idx < 0 {
			continue
		}
		ops[idx].Output = math.MaxInt32
		if res := CheckWithBudget(NewRegisterModel(), FromOperations(ops), unlimited); res.Status != NotLinearizable {
			t.Fatalf("seed %d: corrupted read accepted (%s)\n%s", seed, res.Status, res)
		}
	}
}

// ---------------------------------------------------------------------------
// Cross-validation against a definition-level reference.
//
// The real checker is fast because of two optimisations that could each be
// subtly wrong in ways no hand-written example would catch: the linked list that
// encodes real-time ordering implicitly, and the memo key that stores only the
// in-flight window of the linearised set rather than all of it. Neither is
// obviously correct. So both are checked against a brute-force implementation
// that is a direct transcription of the definition — try every ordering, no
// memoisation, no cleverness — on thousands of small random histories.
// ---------------------------------------------------------------------------

// bruteForce decides linearizability the slow, obvious way. It is exponential
// and only usable on tiny histories, which is exactly what makes it a good
// reference: there is nothing in it to get wrong.
func bruteForce(m Model, ops []Operation) bool {
	used := make([]bool, len(ops))

	returnTime := func(op Operation) int64 {
		if op.Pending {
			return pendingReturn
		}
		return op.Return
	}

	var rec func(state State) bool
	rec = func(state State) bool {
		// Everything that returned has been placed; the definition permits
		// discarding the pending remainder.
		done := true
		for i := range ops {
			if !used[i] && !ops[i].Pending {
				done = false
				break
			}
		}
		if done {
			return true
		}
		for i := range ops {
			if used[i] {
				continue
			}
			// Real-time ordering, straight from the definition: nothing still
			// unplaced may have returned before this operation was invoked.
			eligible := true
			for j := range ops {
				if used[j] || j == i {
					continue
				}
				if returnTime(ops[j]) < ops[i].Call {
					eligible = false
					break
				}
			}
			if !eligible {
				continue
			}
			ok, next := m.Step(state, ops[i].Input, ops[i].observedOutput())
			if !ok {
				continue
			}
			used[i] = true
			if rec(next) {
				used[i] = false
				return true
			}
			used[i] = false
		}
		return false
	}
	return rec(m.Init())
}

func TestAgreesWithBruteForceOnRegisters(t *testing.T) {
	rng := rand.New(rand.NewSource(4242))
	var accepted, rejected int

	for iter := 0; iter < 4000; iter++ {
		n := 2 + rng.Intn(6)
		ops := make([]Operation, n)
		for i := range ops {
			call := int64(rng.Intn(20))
			ops[i] = Operation{
				ClientID: rng.Intn(3),
				Call:     call,
				Return:   call + int64(rng.Intn(8)),
				Pending:  rng.Intn(8) == 0,
			}
			// A tiny value domain, so that randomly chosen outputs are consistent
			// often enough for both verdicts to be well represented.
			if rng.Intn(2) == 0 {
				ops[i].Input = Write(rng.Intn(3))
			} else {
				ops[i].Input = Read()
				ops[i].Output = rng.Intn(3)
			}
		}

		want := bruteForce(NewRegisterModel(), ops)
		got := CheckWithBudget(NewRegisterModel(), FromOperations(ops), unlimited)
		if got.OK() != want {
			t.Fatalf("iteration %d: checker says %s, brute force says linearizable=%v\n%s\n%s",
				iter, got.Status, want, FromOperations(ops), got)
		}
		if want {
			accepted++
		} else {
			rejected++
		}
	}

	// A cross-check that only ever saw one verdict would prove very little.
	if accepted < 200 || rejected < 200 {
		t.Fatalf("generated histories were not varied enough: %d accepted, %d rejected", accepted, rejected)
	}
	t.Logf("agreed with brute force on %d linearizable and %d non-linearizable histories", accepted, rejected)
}

// The same cross-check against a model with a PARTIAL operation, where a pending
// operation genuinely has to be discarded rather than merely ordered late. That
// is the case the register model cannot reach.
func TestAgreesWithBruteForceOnPartialOperations(t *testing.T) {
	rng := rand.New(rand.NewSource(1234))
	var accepted, rejected int

	for iter := 0; iter < 4000; iter++ {
		n := 2 + rng.Intn(5)
		ops := make([]Operation, n)
		for i := range ops {
			call := int64(rng.Intn(12))
			input := "lock"
			if rng.Intn(2) == 0 {
				input = "unlock"
			}
			ops[i] = Operation{
				ClientID: rng.Intn(3),
				Input:    input,
				Call:     call,
				Return:   call + int64(rng.Intn(6)),
				Pending:  rng.Intn(4) == 0,
			}
		}

		want := bruteForce(mutexModel{}, ops)
		got := CheckWithBudget(mutexModel{}, FromOperations(ops), unlimited)
		if got.OK() != want {
			t.Fatalf("iteration %d: checker says %s, brute force says linearizable=%v\n%s\n%s",
				iter, got.Status, want, FromOperations(ops), got)
		}
		if want {
			accepted++
		} else {
			rejected++
		}
	}
	if accepted < 200 || rejected < 200 {
		t.Fatalf("generated histories were not varied enough: %d accepted, %d rejected", accepted, rejected)
	}
	t.Logf("agreed with brute force on %d linearizable and %d non-linearizable histories", accepted, rejected)
}

// ---------------------------------------------------------------------------
// Reporting.
// ---------------------------------------------------------------------------

func TestFailureReportIsStableAcrossRuns(t *testing.T) {
	// Map iteration order is randomised in Go, and this package walks maps when
	// it partitions and when it renders a key-value state. A report that moved
	// between runs would make the same bug look like several.
	ops := randomKVHistory(rand.New(rand.NewSource(5)), 60, 4, 4)
	corruptOneGet(t, rand.New(rand.NewSource(6)), ops)
	h := FromOperations(ops)

	first := CheckWithBudget(NewKVModel(), h, unlimited)
	if first.Status != NotLinearizable {
		t.Fatalf("expected a violation, got %s", first.Status)
	}
	for i := 0; i < 20; i++ {
		again := CheckWithBudget(NewKVModel(), h, unlimited)
		if again.Failure.String() != first.Failure.String() {
			t.Fatalf("report changed between runs:\n--- run 0 ---\n%s\n--- run %d ---\n%s",
				first.Failure.String(), i, again.Failure.String())
		}
		if again.Steps != first.Steps {
			t.Fatalf("search did %d steps, previously %d", again.Steps, first.Steps)
		}
	}
}

func TestFailureReportNamesThePartition(t *testing.T) {
	ops := []Operation{
		{ClientID: 0, Input: Put("a", "1"), Output: nil, Call: 0, Return: 10},
		{ClientID: 0, Input: Get("a"), Output: "1", Call: 20, Return: 30},
		{ClientID: 1, Input: Put("b", "2"), Output: nil, Call: 0, Return: 10},
		{ClientID: 1, Input: Get("b"), Output: "", Call: 20, Return: 30},
	}
	res := mustCheck(t, NewKVModel(), ops, NotLinearizable)
	report := res.Failure.String()
	if !strings.Contains(report, `on key "b"`) {
		t.Errorf("report should name the offending key:\n%s", report)
	}
	if strings.Contains(report, `put("a"`) {
		t.Errorf("report should not drag in the unrelated key:\n%s", report)
	}
	if !strings.Contains(report, "(absent)") {
		t.Errorf("report should use the model's own description of the output:\n%s", report)
	}
}

func TestReportMentionsPendingCaveat(t *testing.T) {
	ops := []Operation{
		{ClientID: 0, Input: Write(1), Output: nil, Call: 0, Return: 10},
		{ClientID: 1, Input: Read(), Output: 0, Call: 20, Return: 30},
		{ClientID: 2, Input: Write(3), Call: 21, Pending: true},
	}
	res := mustCheck(t, NewRegisterModel(), ops, NotLinearizable)
	if !strings.Contains(res.Failure.String(), "NoResponse") {
		t.Errorf("a failure involving pending operations should warn about NoResponse handling:\n%s", res.Failure)
	}
}

// ---------------------------------------------------------------------------
// Recording.
// ---------------------------------------------------------------------------

// The recorder is called from concurrent goroutines by construction, so this
// test is really an assertion about -race.
func TestHistoryRecordsConcurrentClients(t *testing.T) {
	var mu sync.Mutex
	value := 0

	h := NewHistory()
	var wg sync.WaitGroup
	for c := 0; c < 8; c++ {
		wg.Add(1)
		go func(client int) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				if i%2 == 0 {
					v := client*100 + i
					h.Do(client, Write(v), func() any {
						mu.Lock()
						defer mu.Unlock()
						value = v
						return nil
					})
				} else {
					h.Do(client, Read(), func() any {
						mu.Lock()
						defer mu.Unlock()
						return value
					})
				}
			}
		}(c)
	}
	wg.Wait()

	if h.Len() != 400 {
		t.Fatalf("recorded %d operations, want 400", h.Len())
	}
	// A mutex-protected int IS a linearizable register, so this must pass. It is
	// the end-to-end test of the whole package: record a real concurrent
	// workload, then judge it.
	if res := CheckWithBudget(NewRegisterModel(), h, unlimited); !res.OK() {
		t.Fatalf("a mutex-protected int is linearizable by construction:\n%s", res)
	}

	ops := h.Operations()
	for i := 1; i < len(ops); i++ {
		if ops[i].Call < ops[i-1].Call {
			t.Fatalf("Operations must be sorted by invocation time")
		}
	}
}

func TestInvocationLeftPendingStaysPending(t *testing.T) {
	h := NewHistory()
	h.Do(0, Write(1), func() any { return nil })
	h.Invoke(1, Write(2)) // the client "crashed" here
	h.Do(0, Read(), func() any { return 1 })

	ops := h.Operations()
	if len(ops) != 3 || !ops[1].Pending {
		t.Fatalf("the abandoned invocation should be recorded as pending: %v", ops)
	}
	res := Check(NewRegisterModel(), h)
	if !res.OK() {
		t.Fatalf("read of 1 with a concurrent crashed write(2) is fine:\n%s", res)
	}
	if res.Pending != 1 {
		t.Fatalf("Pending = %d, want 1", res.Pending)
	}
}

// Do documents that a panic inside the operation leaves it pending, because
// that is the truthful record. Documented behaviour without a test is how
// documentation becomes fiction.
func TestDoLeavesOperationPendingWhenItPanics(t *testing.T) {
	h := NewHistory()
	func() {
		defer func() { _ = recover() }()
		h.Do(0, Write(1), func() any { panic("the client died here") })
	}()
	ops := h.Operations()
	if len(ops) != 1 || !ops[0].Pending {
		t.Fatalf("the operation should still be pending: %v", ops)
	}
}

func TestDoubleReturnPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("returning twice from one invocation should panic")
		}
	}()
	h := NewHistory()
	inv := h.Invoke(0, Read())
	inv.Return(0)
	inv.Return(0)
}

func TestResponseBeforeInvocationPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("an operation that returned before it was invoked should panic")
		}
	}()
	FromOperations([]Operation{{ClientID: 0, Input: Read(), Output: 0, Call: 100, Return: 10}})
}

func TestNonComparablePartitionKeyPanicsClearly(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("a non-comparable partition key should panic")
		}
		if !strings.Contains(fmt.Sprint(r), "not comparable") {
			t.Fatalf("panic should explain the problem, got %v", r)
		}
	}()
	Check(sliceKeyModel{}, FromOperations([]Operation{{Input: []int{1}, Output: nil, Call: 0, Return: 1}}))
}

type sliceKeyModel struct{}

func (sliceKeyModel) Init() State                          { return 0 }
func (sliceKeyModel) Equal(a, b State) bool                { return a.(int) == b.(int) }
func (sliceKeyModel) Step(s State, _, _ any) (bool, State) { return true, s }
func (sliceKeyModel) PartitionKey(input any) any           { return input }

// ---------------------------------------------------------------------------
// Model plumbing.
// ---------------------------------------------------------------------------

func TestFromDeterministicHandlesNoResponse(t *testing.T) {
	m := NewRegisterModel()
	ok, next := m.Step(3, Write(9), NoResponse)
	if !ok || next.(int) != 9 {
		t.Fatalf("a pending write must be applicable: ok=%v next=%v", ok, next)
	}
	if ok, _ := m.Step(3, Read(), NoResponse); !ok {
		t.Fatal("a pending read must be applicable whatever it would have returned")
	}
	if ok, _ := m.Step(3, Read(), 4); ok {
		t.Fatal("a read that returned the wrong value must be rejected")
	}
}

func TestOptionalInterfacesSurviveTheAdapter(t *testing.T) {
	m := NewKVModel()
	if _, ok := extension[Partitioner](m); !ok {
		t.Fatal("FromDeterministic must not hide the wrapped model's Partitioner")
	}
	if _, ok := extension[Describer](m); !ok {
		t.Fatal("FromDeterministic must not hide the wrapped model's Describer")
	}
	if _, ok := extension[Partitioner](Unpartitioned(m)); ok {
		t.Fatal("Unpartitioned must hide the Partitioner; that is its only job")
	}
	if _, ok := extension[Describer](Unpartitioned(m)); !ok {
		t.Fatal("Unpartitioned must keep failure reports readable")
	}
}

// A state that is mutated in place instead of copied corrupts the memoisation
// table. This asserts the built-in key-value model does not do that.
func TestKVModelDoesNotMutateState(t *testing.T) {
	m := NewKVModel()
	before := map[string]string{"x": "1"}
	if _, next := m.Step(before, Put("x", "2"), nil); next.(map[string]string)["x"] != "2" {
		t.Fatal("put should produce the new value")
	}
	if before["x"] != "1" {
		t.Fatalf("Step mutated the state it was given: %v", before)
	}
	if _, next := m.Step(before, Delete("x"), nil); len(next.(map[string]string)) != 0 {
		t.Fatal("delete should produce an empty state")
	}
	if before["x"] != "1" {
		t.Fatalf("Step mutated the state it was given: %v", before)
	}
}

// ---------------------------------------------------------------------------
// History generation, shared by the property tests and the benchmarks.
// ---------------------------------------------------------------------------

// randomLinearizableHistory builds a history that IS linearizable, and whose
// witnessing order is deliberately NOT the order the operations appear in.
//
// That last part is what makes these histories worth checking. Laying the
// operations out in time order and generating outputs in that same order
// produces a history the checker solves greedily on the first attempt, which
// measures nothing and proves less. Instead this picks a random ordering that
// real time permits, generates the outputs from THAT, and hands back a history
// whose explanation the checker has to go and find.
//
// The intervals are fixed rather than random: each operation is centred on its
// own slot and stretched to overlap its neighbours, so the concurrency width is
// exactly `clients` and a benchmark's cost is attributable.
func randomLinearizableHistory(rng *rand.Rand, spec Deterministic, inputs []any, clients int) []Operation {
	n := len(inputs)
	span := int64(5 * (clients - 1))
	ops := make([]Operation, n)
	for i := range ops {
		centre := int64(i) * 10
		call := centre - span
		if call < 0 {
			call = 0
		}
		ops[i] = Operation{
			ClientID: i % clients,
			Input:    inputs[i],
			Call:     call,
			Return:   centre + span,
		}
	}
	state := spec.Init()
	for _, idx := range randomLinearization(rng, ops) {
		out, next := spec.Apply(state, ops[idx].Input)
		ops[idx].Output = out
		state = next
	}
	return ops
}

// randomLinearization returns a uniformly chosen position for each operation in
// some ordering that respects real time.
//
// An operation may go next exactly when nothing still unplaced returned before
// it was invoked. The two-pointer walk relies on the invocation and response
// times both being non-decreasing in index, which is how the generator above
// lays them out.
func randomLinearization(rng *rand.Rand, ops []Operation) []int {
	n := len(ops)
	placed := make([]bool, n)
	order := make([]int, 0, n)
	lo, hi := 0, 0
	for len(order) < n {
		for lo < n && placed[lo] {
			lo++
		}
		minRet := ops[lo].Return
		if hi <= lo {
			hi = lo + 1
		}
		for hi < n && ops[hi].Call <= minRet {
			hi++
		}
		count := 0
		for i := lo; i < hi; i++ {
			if !placed[i] {
				count++
			}
		}
		pick := rng.Intn(count)
		for i := lo; i < hi; i++ {
			if placed[i] {
				continue
			}
			if pick == 0 {
				placed[i] = true
				order = append(order, i)
				break
			}
			pick--
		}
	}
	return order
}

func randomRegisterHistory(rng *rand.Rand, n, clients int) []Operation {
	inputs := make([]any, n)
	for i := range inputs {
		switch rng.Intn(3) {
		case 0:
			inputs[i] = Write(rng.Intn(8))
		case 1:
			inputs[i] = CAS(rng.Intn(8), rng.Intn(8))
		default:
			inputs[i] = Read()
		}
	}
	return randomLinearizableHistory(rng, registerModel{}, inputs, clients)
}

func randomKVHistory(rng *rand.Rand, n, keys, clients int) []Operation {
	inputs := make([]any, n)
	for i := range inputs {
		key := fmt.Sprintf("k%d", rng.Intn(keys))
		switch rng.Intn(4) {
		case 0:
			inputs[i] = Put(key, fmt.Sprintf("v%d", rng.Intn(6)))
		case 1:
			inputs[i] = Delete(key)
		default:
			inputs[i] = Get(key)
		}
	}
	return randomLinearizableHistory(rng, kvModel{}, inputs, clients)
}

// corruptOneGet changes one read to a value that is never written anywhere in
// the history, so that no ordering at all can produce it. Corrupting to a value
// that IS written elsewhere would be a weaker test: the history might well still
// be linearizable by a different ordering, and a rejection would then be the
// bug.
func corruptOneGet(t *testing.T, rng *rand.Rand, ops []Operation) {
	t.Helper()
	var gets []int
	for i := range ops {
		if in, ok := ops[i].Input.(KVInput); ok && in.Op == OpGet && !ops[i].Pending {
			gets = append(gets, i)
		}
	}
	if len(gets) == 0 {
		t.Skip("generated history contains no reads to corrupt")
	}
	ops[gets[rng.Intn(len(gets))]].Output = "value-nobody-ever-wrote"
}

// ---------------------------------------------------------------------------
// Benchmarks. Every performance claim in this package's documentation comes
// from these.
// ---------------------------------------------------------------------------

// benchCheck times a check and reports the two numbers that explain its cost:
// the concurrency width the search actually faced, and how many model steps it
// took. Both are reported after the loop, because ResetTimer discards
// user-reported metrics.
func benchCheck(b *testing.B, m Model, ops []Operation) {
	h := FromOperations(ops)
	res := CheckWithBudget(m, h, unlimited)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if r := CheckWithBudget(m, h, unlimited); r.Status != res.Status {
			b.Fatalf("unstable verdict: %s then %s", res.Status, r.Status)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(res.Widest), "width")
	b.ReportMetric(float64(res.Steps), "steps")
}

// Cost against history length with no concurrency at all: the floor.
func BenchmarkLengthSequential(b *testing.B) {
	for _, n := range []int{100, 1000, 10000, 100000} {
		b.Run(fmt.Sprintf("ops=%d", n), func(b *testing.B) {
			ops := randomRegisterHistory(rand.New(rand.NewSource(1)), n, 1)
			benchCheck(b, NewRegisterModel(), ops)
		})
	}
}

// Cost against history length at a realistic width. This is the axis that
// scales.
func BenchmarkLengthConcurrent(b *testing.B) {
	for _, n := range []int{100, 400, 1600, 6400} {
		b.Run(fmt.Sprintf("ops=%d", n), func(b *testing.B) {
			ops := randomRegisterHistory(rand.New(rand.NewSource(2)), n, 4)
			benchCheck(b, NewRegisterModel(), ops)
		})
	}
}

// Cost against concurrency width at a fixed length. This is the axis that does
// not scale, and the reason the documentation tells you to partition.
//
// It stops at 14 clients because 16 already takes about a second per check on
// this machine and 20 takes minutes — which is itself the measurement the
// documentation quotes, and the reason a benchmark cannot go there.
func BenchmarkWidth(b *testing.B) {
	for _, clients := range []int{2, 4, 6, 8, 10, 12, 14} {
		b.Run(fmt.Sprintf("clients=%d", clients), func(b *testing.B) {
			ops := randomRegisterHistory(rand.New(rand.NewSource(3)), 100, clients)
			benchCheck(b, NewRegisterModel(), ops)
		})
	}
}

// Refuting a broken history is the expensive direction: finding one witness can
// be lucky, proving none exists cannot.
func BenchmarkWidthRefute(b *testing.B) {
	for _, clients := range []int{2, 4, 6, 8, 10, 12} {
		b.Run(fmt.Sprintf("clients=%d", clients), func(b *testing.B) {
			ops := randomRegisterHistory(rand.New(rand.NewSource(4)), 100, clients)
			for i := len(ops) - 1; i >= 0; i-- {
				if ops[i].Input.(RegisterInput).Op == OpRead {
					ops[i].Output = math.MaxInt32
					break
				}
			}
			benchCheck(b, NewRegisterModel(), ops)
		})
	}
}

// The headline optimisation, measured both ways on the same history.
func BenchmarkPartitioned(b *testing.B) {
	for _, n := range []int{200, 500, 1000, 2000} {
		ops := randomKVHistory(rand.New(rand.NewSource(5)), n, 100, 8)
		b.Run(fmt.Sprintf("ops=%d/partitioned", n), func(b *testing.B) {
			benchCheck(b, NewKVModel(), ops)
		})
		b.Run(fmt.Sprintf("ops=%d/whole", n), func(b *testing.B) {
			benchCheck(b, Unpartitioned(NewKVModel()), ops)
		})
	}
}
