package sim

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"runtime"
	"strings"
	"time"
)

// EventKind classifies a trace entry.
type EventKind uint8

const (
	// EventSchedule records the scheduler handing the baton to an actor. These
	// are the decisions that define the run: two traces with identical schedule
	// events explored the same interleaving.
	EventSchedule EventKind = iota
	// EventYield records an explicit Sim.Yield.
	EventYield
	// EventSleep records an actor going to sleep on the virtual clock.
	EventSleep
	// EventClockAdvance records virtual time jumping forward.
	EventClockAdvance
	// EventSend, EventRecv record completed channel operations.
	EventSend
	EventRecv
	// EventBlockSend, EventBlockRecv record an actor parking on a channel.
	EventBlockSend
	EventBlockRecv
	// EventCrash records an actor being killed.
	EventCrash
	// EventNet records a network-level action: delivery, drop, delay, partition.
	EventNet
	// EventUser is emitted by Sim.Record, for anything a test wants in the
	// trace.
	EventUser
)

var eventNames = [...]string{
	EventSchedule:     "sched",
	EventYield:        "yield",
	EventSleep:        "sleep",
	EventClockAdvance: "clock",
	EventSend:         "send",
	EventRecv:         "recv",
	EventBlockSend:    "block-send",
	EventBlockRecv:    "block-recv",
	EventCrash:        "crash",
	EventNet:          "net",
	EventUser:         "user",
}

func (k EventKind) String() string {
	if int(k) < len(eventNames) && eventNames[k] != "" {
		return eventNames[k]
	}
	return fmt.Sprintf("kind(%d)", uint8(k))
}

// Event is one entry in a trace.
type Event struct {
	Step   int
	Time   time.Time
	Actor  string
	Kind   EventKind
	Detail string
}

func (e Event) String() string {
	rel := e.Time.Sub(DefaultStartTime)
	if e.Detail == "" {
		return fmt.Sprintf("%6d %12s %-12s %s", e.Step, rel, e.Actor, e.Kind)
	}
	return fmt.Sprintf("%6d %12s %-12s %-11s %s", e.Step, rel, e.Actor, e.Kind, e.Detail)
}

// Trace is the recorded history of a run.
//
// # Why this exists
//
// A trace turns the determinism guarantee from a claim into something testable.
// Run the same seed twice, hash both traces, compare: identical hashes mean the
// two runs really did explore the same schedule, down to the last decision.
// Without it, "deterministic" is an assertion nobody can check — and a simulator
// whose determinism is unverified will eventually stop being deterministic
// without anyone noticing.
//
// The second use is diagnosis. When a run fails, the trace is the story of how
// it got there: which actor ran, in what order, when time moved, which messages
// were dropped. Reading the last twenty events usually explains the failure.
type Trace struct {
	events []Event
}

func (t *Trace) add(e Event) {
	if t == nil {
		return
	}
	t.events = append(t.events, e)
}

// Events returns the recorded events.
func (t *Trace) Events() []Event {
	if t == nil {
		return nil
	}
	return t.events
}

// Len returns the event count.
func (t *Trace) Len() int {
	if t == nil {
		return 0
	}
	return len(t.events)
}

// Hash returns a stable digest of the whole trace.
//
// This is the determinism check in one value. It covers the step, the virtual
// time, the actor and the kind of every event — everything that defines a
// schedule — so two runs agreeing on it agree on the run.
//
// Detail is included too. It carries message identities and drop reasons, so
// omitting it would let two runs that delivered different messages in the same
// order hash identically, which is precisely the kind of near-miss a
// determinism test exists to catch.
func (t *Trace) Hash() string {
	h := sha256.New()
	if t != nil {
		for _, e := range t.events {
			fmt.Fprintf(h, "%d|%d|%s|%d|%s\n",
				e.Step, e.Time.UnixNano(), e.Actor, e.Kind, e.Detail)
		}
	}
	return hex.EncodeToString(h.Sum(nil)[:8])
}

// String renders the whole trace, one event per line.
func (t *Trace) String() string {
	if t == nil {
		return "(tracing disabled)"
	}
	var b strings.Builder
	for _, e := range t.events {
		b.WriteString(e.String())
		b.WriteByte('\n')
	}
	return b.String()
}

// Tail renders the last n events, which is what you actually want after a
// failure — the beginning of a long run is rarely where the bug is.
func (t *Trace) Tail(n int) string {
	if t == nil {
		return "(tracing disabled)"
	}
	start := len(t.events) - n
	if start < 0 {
		start = 0
	}
	var b strings.Builder
	if start > 0 {
		fmt.Fprintf(&b, "... %d earlier events omitted ...\n", start)
	}
	for _, e := range t.events[start:] {
		b.WriteString(e.String())
		b.WriteByte('\n')
	}
	return b.String()
}

// Record adds a user event to the trace.
//
// Use it to mark domain milestones — "leader elected", "write acknowledged" —
// so a trace reads as a story about your system rather than a list of
// scheduler decisions. Recorded details participate in the trace hash, so a
// run that reached different milestones is detectably different.
func (s *Sim) Record(format string, args ...any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.trace == nil {
		return
	}
	name := ""
	if s.running >= 0 && s.running < len(s.actors) {
		name = s.actors[s.running].name
	}
	s.trace.add(Event{
		Step:   s.steps,
		Time:   s.clock.nowLocked(),
		Actor:  name,
		Kind:   EventUser,
		Detail: fmt.Sprintf(format, args...),
	})
}

// stack captures the panicking goroutine's stack for the failure report.
//
// runtime.Stack(buf, false) rather than true: the panicking actor's own stack is
// what identifies the bug, while every other goroutine in the process is parked
// in the scheduler and would add pages of identical noise to every report.
func stack() []byte {
	buf := make([]byte, 8192)
	n := runtime.Stack(buf, false)
	return buf[:n]
}
