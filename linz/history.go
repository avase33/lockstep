package linz

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
)

// Operation is one invocation and its response: what a client asked for, when
// it asked, what it got back, and when.
//
// The two timestamps are the reason this type exists rather than a plain list of
// calls. Linearizability is defined in terms of real-time ordering — if this
// operation's response happened before another's invocation, no valid
// explanation may order them the other way — and without both endpoints that
// constraint cannot be expressed. A history that records only "these operations
// happened, in roughly this order" can be checked for sequential consistency and
// nothing stronger.
type Operation struct {
	// ClientID identifies the goroutine, connection or process that issued the
	// operation. The checker never constrains operations by client; the field
	// exists so failure reports can say who did what, which is usually the first
	// thing you want to know.
	ClientID int

	// Input is what was requested, in whatever form the Model understands.
	Input any

	// Output is what came back. Ignored when Pending is true.
	Output any

	// Call is the time of the invocation and Return the time of the response,
	// on any monotonically increasing scale you like: nanoseconds from
	// time.Now, a simulator's virtual clock, or a plain counter. Only the
	// relative order matters, and only across operations — the numbers are never
	// interpreted as durations.
	//
	// Two events with equal timestamps are treated as concurrent. That is the
	// safe direction: it lets the checker consider more orderings, so a coarse
	// clock can cause a violation to be missed but can never invent one.
	Call   int64
	Return int64

	// Pending marks an operation that was invoked and never returned, because
	// the client crashed, the connection dropped, or the test simply stopped
	// waiting. Its Output and Return are ignored.
	//
	// These are not an edge case to be tidied away — they are the operations
	// that make crash testing worth doing. See the package documentation.
	Pending bool
}

// pendingReturn is the response time the checker assigns to an operation that
// never returned: later than every real timestamp, so nothing is ever forced to
// be ordered before it.
const pendingReturn int64 = math.MaxInt64

// String renders one operation for a report or a debug dump.
func (op Operation) String() string {
	ret := fmt.Sprintf("%d", op.Return)
	if op.Pending {
		ret = "never"
	}
	return fmt.Sprintf("[client %d] %s (invoked %d, returned %s)",
		op.ClientID, defaultDescribeOperation(op.Input, op.observedOutput()), op.Call, ret)
}

// observedOutput is the output the checker should reason about: the recorded one
// normally, and NoResponse for an operation whose client never saw a response.
func (op Operation) observedOutput() any {
	if op.Pending {
		return NoResponse
	}
	return op.Output
}

// History is a recording of everything the clients did, and the thing the
// checker consumes.
//
// It is safe for concurrent use, which is the whole point: the natural way to
// produce a history is to run a genuinely concurrent test — several goroutines
// hammering the system under test, ideally under -race — and have each of them
// record its own operations as it goes. A recorder that needed external locking
// would either be wrong or would serialise the very concurrency the test is
// trying to create.
//
// # About the timestamps
//
// By default History timestamps events with its own monotonic counter,
// incremented under the same lock that appends the event. Two consequences
// follow, and both are wanted. The counter never goes backwards, so the recorded
// order is always a legal witness of real time; and it costs no syscall, so
// recording perturbs the timing of the system under test far less than calling
// time.Now would.
//
// Because Invoke is called just before the operation starts and Return just
// after it finishes, the recorded interval is always at least as wide as the
// real one. Wider intervals mean more apparent concurrency, which means the
// checker considers more orderings. Again the safe direction: sloppy recording
// can hide a violation, never manufacture one.
//
// If you have real timestamps — from a simulator's virtual clock, or from a
// server-side log — use Add and supply them yourself.
type History struct {
	mu    sync.Mutex
	ops   []Operation
	clock int64
}

// NewHistory returns an empty history ready to record.
func NewHistory() *History { return &History{} }

// FromOperations builds a history from operations you already have — replayed
// from a log file, decoded from a trace, or constructed literally in a test.
//
// The operations are copied, so the caller's slice can be reused, and the
// history's own counter is advanced past their timestamps so that recording more
// operations onto the end with Invoke continues rather than collides.
func FromOperations(ops []Operation) *History {
	h := NewHistory()
	for _, op := range ops {
		h.Add(op)
	}
	return h
}

// Invocation is an operation that has started and not yet returned.
//
// Holding one is how a pending operation gets recorded: an Invocation whose
// Return is never called stays in the history as an operation the checker must
// consider both ways. There is no "abandon" method because there is nothing to
// call it from — the client that would have called it is the one that crashed.
type Invocation struct {
	h   *History
	idx int
}

// Invoke records that a client has just issued an operation, and returns a
// handle for recording the response.
//
// Call it immediately before the call into the system under test, and call
// Return immediately after. Anything you do between those two points is inside
// the operation's recorded interval, which is why the pair should bracket the
// call as tightly as possible: a wide interval is not incorrect, but it gives
// the checker more freedom and so weakens the test.
func (h *History) Invoke(clientID int, input any) *Invocation {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.clock++
	h.ops = append(h.ops, Operation{
		ClientID: clientID,
		Input:    input,
		Call:     h.clock,
		Pending:  true,
	})
	return &Invocation{h: h, idx: len(h.ops) - 1}
}

// Return records the response to an invocation.
//
// Calling it twice panics. That is a harness bug — an operation has one
// response — and a silently overwritten output would produce a history that
// disagrees with what the system actually did, which makes every later verdict
// worthless.
func (inv *Invocation) Return(output any) {
	h := inv.h
	h.mu.Lock()
	defer h.mu.Unlock()
	op := &h.ops[inv.idx]
	if !op.Pending {
		panic(fmt.Sprintf("linz: operation %v already returned; Return must be called exactly once", op.Input))
	}
	h.clock++
	op.Output = output
	op.Return = h.clock
	op.Pending = false
}

// Do records a complete operation around a call into the system under test.
//
// It is the form to prefer when the client does not crash, because it makes the
// invocation and the response impossible to mismatch:
//
//	h.Do(clientID, linz.Read(), func() any { return store.Read() })
//
// If f panics the operation is left pending, which is the truthful record: the
// call was made and no response was ever observed. The panic propagates.
func (h *History) Do(clientID int, input any, f func() any) {
	inv := h.Invoke(clientID, input)
	inv.Return(f())
}

// Add appends an operation with timestamps you supply.
//
// Use it when the times come from somewhere better than this package's counter —
// a deterministic simulator's virtual clock, or a server's own log — and when
// reconstructing a history in a test where the timing is the thing under test.
func (h *History) Add(op Operation) {
	validate(op)
	h.mu.Lock()
	defer h.mu.Unlock()
	h.ops = append(h.ops, op)
	if op.Call > h.clock {
		h.clock = op.Call
	}
	if !op.Pending && op.Return > h.clock {
		h.clock = op.Return
	}
}

// Len returns the number of recorded operations, pending ones included.
func (h *History) Len() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.ops)
}

// Operations returns a copy of the history, sorted by invocation time.
//
// Sorted so that a printed history reads in the order things happened, and so
// that two runs of the same test produce byte-identical output. The checker does
// its own ordering and does not depend on this.
func (h *History) Operations() []Operation {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]Operation, len(h.ops))
	copy(out, h.ops)
	// Stable, with an explicit tiebreak on client id, so operations recorded at
	// the same instant do not swap places between runs.
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Call != out[j].Call {
			return out[i].Call < out[j].Call
		}
		return out[i].ClientID < out[j].ClientID
	})
	return out
}

// String renders the whole history, one operation per line. Useful in a t.Log
// when a verdict surprises you.
func (h *History) String() string {
	var b strings.Builder
	for _, op := range h.Operations() {
		b.WriteString(op.String())
		b.WriteByte('\n')
	}
	return b.String()
}

// validate rejects histories that cannot have happened.
//
// It panics rather than returning an error because a response that precedes its
// own invocation is a bug in the recording harness, not a finding about the
// system under test, and quietly checking a nonsensical history would produce a
// verdict about nothing. Failing at the point the bad operation is added names
// the culprit; failing later would not.
func validate(op Operation) {
	if !op.Pending && op.Return < op.Call {
		panic(fmt.Sprintf("linz: operation %v returned at %d but was invoked at %d; "+
			"a response cannot precede its invocation", op.Input, op.Return, op.Call))
	}
}
