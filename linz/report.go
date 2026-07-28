package linz

import (
	"fmt"
	"strings"
	"time"
)

// Status is a checker verdict. There are three, not two, and the third is the
// important one.
type Status uint8

const (
	// Linearizable means the checker found an ordering of every operation that
	// respects real time and reproduces every observed output. The history is
	// consistent with a correct implementation.
	//
	// It is a statement about this history only. Linearizability is not
	// something a single run can establish; it is something a single run can
	// refute. Run many seeds.
	Linearizable Status = iota

	// NotLinearizable means no such ordering exists. This is a proof, not a
	// heuristic: the search is exhaustive, so if it says no, the system under
	// test (or the model, or the recording harness) is wrong. Result.Failure
	// says where.
	NotLinearizable

	// Unknown means the search ran out of budget before it could decide.
	//
	// This exists as a distinct verdict because the alternative — treating a
	// timeout as a pass — would make the whole package worthless. A checker that
	// says "linearizable" when it means "I gave up" turns every hard history,
	// which is to say every interesting one, into a green test. Treat Unknown as
	// a failure in CI, then either raise the budget or reduce the concurrency of
	// the workload until the check completes.
	Unknown
)

func (s Status) String() string {
	switch s {
	case Linearizable:
		return "linearizable"
	case NotLinearizable:
		return "not linearizable"
	case Unknown:
		return "unknown"
	}
	return fmt.Sprintf("Status(%d)", uint8(s))
}

// Result is the outcome of a check.
//
// It is a value rather than an error because a successful check still has things
// worth looking at — how many partitions there were, how much of the budget the
// search burned — and because the counts are what tell you whether a green
// result came from a workload that was actually exercising anything.
type Result struct {
	// Status is the verdict. Check it explicitly; do not test Failure for nil,
	// because Unknown has no Failure either.
	Status Status

	// Operations is how many operations were checked, and Pending how many of
	// those never returned.
	Operations int
	Pending    int

	// Partitions is how many independent sub-problems the history was split
	// into. A number stuck at 1 on a key-value workload means the Partitioner is
	// missing or is returning a constant, which is the difference between a
	// check that finishes and one that does not.
	Partitions int

	// Largest is the size of the biggest partition, and Widest the greatest
	// number of operations in flight simultaneously within any partition.
	//
	// Widest is the number to watch: search cost is exponential in it. If a
	// check is slow or comes back Unknown, this is the field that explains why,
	// and reducing it — fewer clients per key — is the fix that works.
	Largest int
	Widest  int

	// Steps is how many model transitions the search evaluated, and Elapsed how
	// long it took. Both are reported on success too, so a check that is
	// creeping towards its budget can be noticed before it starts failing.
	Steps   int64
	Elapsed time.Duration

	// Failure describes the violation. Non-nil exactly when Status is
	// NotLinearizable.
	Failure *Failure

	// Reason explains an Unknown verdict.
	Reason string
}

// OK reports whether the history was proved linearizable. Note that Unknown is
// not OK — see the Unknown documentation for why that matters.
func (r Result) OK() bool { return r.Status == Linearizable }

// String renders the whole result, including the failure report. This is what
// belongs in a t.Fatal.
func (r Result) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: %d operations", r.Status, r.Operations)
	if r.Pending > 0 {
		fmt.Fprintf(&b, " (%d never returned)", r.Pending)
	}
	fmt.Fprintf(&b, " in %d partition", r.Partitions)
	if r.Partitions != 1 {
		b.WriteByte('s')
	}
	fmt.Fprintf(&b, " (largest %d, max concurrency %d); %d model steps in %s",
		r.Largest, r.Widest, r.Steps, r.Elapsed.Round(time.Microsecond))

	switch r.Status {
	case Unknown:
		fmt.Fprintf(&b, "\n\n%s\n", r.Reason)
		b.WriteString(unknownAdvice)
	case NotLinearizable:
		if r.Failure != nil {
			b.WriteString("\n\n")
			b.WriteString(r.Failure.String())
		}
	}
	return b.String()
}

const unknownAdvice = `
  The search did not finish, so this history is neither proved correct nor
  proved wrong. Do not treat it as a pass. Either raise the budget
  (CheckWithBudget), or lower the number of operations that overlap in time on a
  single partition, which is what the cost is exponential in.
`

// Failure explains why no ordering exists.
//
// "Not linearizable" on its own tells a reader that something is wrong and
// nothing about what, which leaves them staring at a thousand-line history. The
// fields here answer the three questions they are about to ask: how far did you
// get, what did the object look like there, and what could not happen next.
type Failure struct {
	// PartitionKey is the key whose sub-history could not be explained, when the
	// model partitions. Everything below concerns that partition only.
	PartitionKey any
	Partitioned  bool

	// Prefix is the longest sequence of operations that CAN be linearised, in
	// the order the checker put them. Every one of these is consistent; the
	// violation is at the boundary that follows.
	Prefix []Operation

	// State is the model's state after Prefix. This is the value the blocked
	// operations were checked against, and usually the fact that identifies the
	// bug on sight.
	State State

	// Blocked lists the operations that were eligible to come next and could
	// not. These are the operations that had already been invoked and had not
	// yet returned at that point in the ordering, so one of them had to be next.
	Blocked []Operation

	// BlockedRejected parallels Blocked: true where the model ruled the
	// operation impossible in State, false where the branch was pruned because
	// an equivalent search state had already been explored and led nowhere.
	// Distinguishing them matters — the first kind is the violation, the second
	// is only an artefact of the search.
	BlockedRejected []bool

	// Barrier is the operation whose response ended the search: everything not
	// yet placed was invoked after this returned, so nothing else was allowed to
	// come next. Nil if the search ran out of operations instead.
	Barrier *Operation

	// Total is the number of operations in the failing partition.
	Total int

	describe      func(input, output any) string
	describeState func(State) string
}

func (f *Failure) describeOp(op Operation) string {
	if f.describe != nil {
		return f.describe(op.Input, op.observedOutput())
	}
	return defaultDescribeOperation(op.Input, op.observedOutput())
}

func (f *Failure) stateString() string {
	if f.describeState != nil {
		return f.describeState(f.State)
	}
	return fmt.Sprintf("%v", f.State)
}

// String renders the report. The shape is deliberate: the reader is told what
// worked, then what the world looked like, then exactly what could not happen,
// then why nothing else was allowed to happen instead.
func (f *Failure) String() string {
	var b strings.Builder

	b.WriteString("no ordering of these operations can explain what the clients observed")
	if f.Partitioned {
		fmt.Fprintf(&b, "\non key %v (%d operations touch it)", quoteKey(f.PartitionKey), f.Total)
	}
	b.WriteString(".\n\n")

	fmt.Fprintf(&b, "  the longest ordering that works covers %d of %d operations:\n",
		len(f.Prefix), f.Total)
	if len(f.Prefix) == 0 {
		b.WriteString("      (none: the very first operation could not be placed)\n")
	}
	for i, op := range f.Prefix {
		fmt.Fprintf(&b, "   %3d. %s   [client %d, invoked %d, returned %s]\n",
			i+1, f.describeOp(op), op.ClientID, op.Call, returnString(op))
	}

	fmt.Fprintf(&b, "\n  state after those operations: %s\n", f.stateString())

	fmt.Fprintf(&b, "\n  one of these had to happen next, and none of them could:\n")
	for i, op := range f.Blocked {
		why := "already explored from an equivalent state, and led nowhere"
		if i < len(f.BlockedRejected) && f.BlockedRejected[i] {
			why = "impossible in the state above"
		}
		fmt.Fprintf(&b, "      %s   [client %d, invoked %d, returned %s]\n",
			f.describeOp(op), op.ClientID, op.Call, returnString(op))
		fmt.Fprintf(&b, "          %s\n", why)
	}

	// The barrier clause only says something when there ARE other unplaced
	// operations for it to be talking about. When everything unplaced is already
	// listed above, it is noise, and a report nobody reads to the end is a report
	// that does not work.
	if f.Barrier != nil && f.Total-len(f.Prefix) > len(f.Blocked) {
		fmt.Fprintf(&b, "\n  and nothing else was eligible: %s returned at %d, and every\n"+
			"  operation not listed above was invoked only after that, so none of them\n"+
			"  can have taken effect first.\n",
			f.describeOp(*f.Barrier), f.Barrier.Return)
	}

	if hasPending(f.Prefix) || hasPending(f.Blocked) {
		b.WriteString(pendingNote)
	}
	return b.String()
}

const pendingNote = `
  This history contains operations that never returned. If you implemented Model
  yourself rather than using FromDeterministic, check that Step treats
  linz.NoResponse as "any output is acceptable" — a model that does not will
  refuse to place those operations and can produce this report on a history that
  is perfectly fine.
`

func hasPending(ops []Operation) bool {
	for _, op := range ops {
		if op.Pending {
			return true
		}
	}
	return false
}

func returnString(op Operation) string {
	if op.Pending {
		return "never"
	}
	return fmt.Sprintf("%d", op.Return)
}

func quoteKey(k any) string {
	if s, ok := k.(string); ok {
		return fmt.Sprintf("%q", s)
	}
	return fmt.Sprintf("%v", k)
}
