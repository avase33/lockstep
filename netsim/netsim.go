// Package netsim is a simulated network for lockstep: nodes, asynchronous
// message delivery, and the faults that make distributed systems hard.
//
// # What this is for
//
// A distributed system is mostly correct until the network stops being polite.
// Messages arrive late, arrive twice, arrive out of order, or never arrive at
// all; two halves of a cluster stop hearing each other; a node dies with
// messages in flight toward it. Every one of those is rare in a test lab and
// routine in production, which is exactly the wrong way round.
//
// netsim makes them ordinary. Faults are configured as probabilities and drawn
// from the simulation's seeded PRNG, so a run that partitions the cluster at a
// pathological moment is not luck — it is a seed, and it can be replayed until
// the bug is understood and then kept in CI forever.
//
// # The determinism contract
//
// This package is worthless if it is not deterministic, so it obeys the rules in
// package sim without exception:
//
//   - Every random choice comes from Sim.Rand. There is no math/rand import here
//     and there must never be one.
//   - Every wait goes through Clock.Sleep. Delivery latency is virtual time, so
//     a simulated ten-second network delay costs microseconds of real time.
//   - No map is ever iterated in a way that affects behaviour. Where a map must
//     be walked, its keys are sorted first. See Network.Nodes and Network.Close.
//   - Trace details are built only from names, sequence numbers and durations —
//     never from a caller's message body. Formatting a body with %v could print
//     a pointer address, and a pointer address in the trace hash would make the
//     determinism test fail at random for reasons that look nothing like the
//     cause. This is a real trap and it is why Message.Body never reaches the
//     trace.
//
// # The delivery model
//
// Send never blocks. It decides the message's fate immediately — dropped,
// delayed, duplicated — and then hands each surviving copy to a courier: a
// short-lived actor that sleeps for the drawn latency and deposits the message
// in the destination's inbox.
//
// One courier per copy, rather than one delivery actor with a sorted queue. The
// queue design is tempting and slightly cheaper, but it needs to abandon a sleep
// when a message arrives that is due sooner, and package sim's Clock offers no
// way to cancel or shorten a pending timer. Polling around that would coarsen
// virtual time for everyone. Clock.After already spawns an actor per timer, so
// per-copy couriers are the idiom here rather than an exception to it.
//
// The cost is honest: an in-flight message is a registered actor, and Sim's
// scheduler scans every actor it has ever created on each step. A run pushing
// hundreds of thousands of messages will feel that. Runs of that size are
// better expressed as several shorter runs over different seeds anyway, which is
// also how you get more schedule coverage per second.
//
// # Ordering
//
// There is no ordering guarantee, by construction. Each copy draws its own
// latency independently, so two messages sent back to back routinely arrive
// backwards. That is not a flaw being tolerated; it is the single most
// productive fault this package injects, because code that assumes ordering
// usually looks correct right up to the moment it is not.
package netsim

import (
	"fmt"
	"sort"
	"time"

	"github.com/avase33/lockstep/sim"
)

// DefaultInboxCapacity bounds how many delivered-but-unread messages a node may
// hold before the network starts dropping.
//
// A bound is not optional. An unbounded inbox turns a node that has stopped
// reading into a memory leak that a test will never notice, and a zero-capacity
// inbox would park every courier until the receiver happened to be listening,
// which converts a slow reader into a simulation-wide deadlock. Overflow is
// reported as a drop with reason "inbox-full" so that a receiver falling behind
// shows up in the trace as what it is, rather than as messages that mysteriously
// went missing.
const DefaultInboxCapacity = 128

// Config describes the network's behaviour. The zero value is a perfect
// network: no loss, no duplication, no delay.
//
// Perfect-by-default is deliberate, for the same reason sim.Config phrases its
// tracing switch negatively. A test that forgets to configure faults should get
// a boring network and a boring result, not silent chaos it never asked for and
// cannot explain. You turn the faults on, one field at a time, and every one you
// turn on is visible in the code that turned it on.
type Config struct {
	// MinLatency and MaxLatency bound the delay drawn for each copy, inclusive
	// at both ends. Inclusive at the top because a caller writing
	// MaxLatency = timeout means "let a message land exactly on the deadline",
	// and that boundary is where timeout bugs live.
	//
	// The two are drawn per copy, not per message and not per link, which is
	// what makes reordering and duplicate-overtakes-original happen on their own
	// rather than needing a separate fault knob.
	MinLatency time.Duration
	MaxLatency time.Duration

	// DropRate is the probability in [0,1] that a message is discarded at send
	// time. Values outside the range are clamped rather than rejected, matching
	// Rand.Bool: fault rates are often computed by scaling a base rate, and a
	// value that lands at 1.0000000001 should mean "always", not "fail the run".
	DropRate float64

	// DuplicateRate is the probability in [0,1] that a message is delivered
	// twice, each copy with its own independently drawn latency.
	//
	// Independent latency matters more than the duplication itself. Two copies
	// arriving back to back are easy to deduplicate by accident; a copy arriving
	// a full timeout after the original, interleaved with the retry it provoked,
	// is what actually breaks at-most-once handling.
	DuplicateRate float64

	// InboxCapacity is the per-node receive buffer. Defaults to
	// DefaultInboxCapacity when not positive.
	InboxCapacity int
}

func (c Config) withDefaults() Config {
	if c.InboxCapacity <= 0 {
		c.InboxCapacity = DefaultInboxCapacity
	}
	// A negative latency would schedule a delivery before its own send, which is
	// not a fault worth modelling — it is nonsense, and it would make the
	// causality invariant untestable. Clamping is the only repair that keeps the
	// rest of the config meaningful.
	if c.MinLatency < 0 {
		c.MinLatency = 0
	}
	if c.MaxLatency < 0 {
		c.MaxLatency = 0
	}
	// Normalise an inverted range once, here, instead of relying on Rand.Duration
	// to swap on every draw. Doing it at construction means Config.MaxLatency read
	// back later says what the network will actually do.
	if c.MaxLatency < c.MinLatency {
		c.MinLatency, c.MaxLatency = c.MaxLatency, c.MinLatency
	}
	return c
}

// Stats counts what the network did. All counts are cumulative for the life of
// the Network.
type Stats struct {
	// Sent is the number of Send calls that named a valid pair of nodes,
	// including those the network immediately discarded. It counts intent, not
	// outcome, which is what makes the identity in Balanced meaningful.
	Sent uint64

	// Delivered counts copies placed in a destination inbox. A duplicated
	// message that arrives twice contributes 2.
	Delivered uint64

	// Dropped counts copies discarded, for any reason: random loss, a partition,
	// a crashed destination, a full inbox, or a closed network.
	Dropped uint64

	// Duplicated counts messages that were chosen for duplication — messages,
	// not copies. One duplicated message adds 1 here and (if both copies land)
	// 2 to Delivered. Counting copies instead would make the accounting identity
	// in Balanced silently untrue, which is precisely the kind of counter bug
	// that invalidates a whole test suite's assertions.
	Duplicated uint64

	// DelayedTotal is the sum of the latencies drawn for every copy actually put
	// on the wire. Copies dropped at send time never draw a latency and so
	// contribute nothing; copies dropped on arrival do contribute, because the
	// wire time was really spent.
	//
	// Divide by Delivered+droppedOnArrival for a mean, or just watch it move to
	// confirm latency injection is switched on at all.
	DelayedTotal time.Duration
}

// Balanced reports whether the counters satisfy the network's accounting
// identity: Sent == Delivered + Dropped - Duplicated.
//
// The identity holds because every Send produces exactly one copy, or two when
// duplication fires, and every copy ends in exactly one of Delivered or Dropped.
// So Delivered+Dropped counts copies, which is Sent+Duplicated.
//
// It is exported rather than left to tests because it is the cheapest possible
// assertion that the fault injector is not losing track of messages, and a
// simulator whose own bookkeeping drifts will happily report a clean run of a
// broken system.
func (s Stats) Balanced() bool {
	return s.Sent+s.Duplicated == s.Delivered+s.Dropped
}

func (s Stats) String() string {
	return fmt.Sprintf("sent=%d delivered=%d dropped=%d duplicated=%d delayed-total=%s",
		s.Sent, s.Delivered, s.Dropped, s.Duplicated, s.DelayedTotal)
}

// Network is a simulated network. Create one with New, add endpoints with Join.
//
// Like sim.Sim, a Network is not safe for use from a goroutine the scheduler
// does not manage, and it needs no internal locking for the same reason: the
// scheduler runs at most one actor at a time, so every method below executes
// while its caller holds the baton, and control passes between actors through
// the scheduler's mutex and hand-off channel. Adding a mutex here would suggest
// a concurrency story that does not exist and would obscure the rule that
// actually keeps this correct.
type Network struct {
	sim *sim.Sim
	cfg Config

	nodes map[string]*Node

	// blocked is the set of directed links the network refuses to carry. It is
	// only ever probed by key; the one place it is enumerated (String) sorts
	// first.
	blocked map[link]bool

	// seq numbers messages network-wide. It exists so a trace line names one
	// specific message rather than "a message from a to b", which is the
	// difference between reading a trace and guessing at one. Network-wide
	// rather than per-link because a single monotonic counter also encodes send
	// order across the whole cluster.
	seq uint64

	closed bool
	stats  Stats
}

// link is one direction of traffic between two nodes. Directed, because
// asymmetric partitions are the interesting ones.
type link struct{ from, to string }

// New creates a network attached to a simulation.
//
// The Network borrows the Sim's PRNG rather than owning one. That is the whole
// point: fault decisions and scheduling decisions must come from the same
// stream, so that a seed pins down the entire run and not merely the half of it
// the scheduler controls.
func New(s *sim.Sim, cfg Config) *Network {
	if s == nil {
		panic("lockstep/netsim: New requires a non-nil *sim.Sim")
	}
	return &Network{
		sim:     s,
		cfg:     cfg.withDefaults(),
		nodes:   make(map[string]*Node),
		blocked: make(map[link]bool),
	}
}

// Config returns the normalised configuration actually in force, after defaults
// and clamping. Worth reading back in a test that computes its fault rates:
// what you configured and what the network does are not always the same thing.
func (nw *Network) Config() Config { return nw.cfg }

// Join adds an endpoint and returns its handle. Names must be unique.
//
// The name must match the sim actor that serves this node, because that is how
// the network learns the node has died: delivery checks Sim.Crashed(name). Two
// separate identities — an actor name and a node name — would let a test crash
// an actor and then watch the network keep cheerfully delivering to it, and the
// resulting "impossible" trace would take an afternoon to explain.
func (nw *Network) Join(name string) *Node {
	if name == "" {
		panic("lockstep/netsim: Join requires a non-empty node name")
	}
	if _, dup := nw.nodes[name]; dup {
		panic(fmt.Sprintf("lockstep/netsim: duplicate node %q; names must be unique so traces are readable", name))
	}
	n := &Node{
		net:   nw,
		name:  name,
		inbox: sim.NewChan(nw.sim, nw.cfg.InboxCapacity),
	}
	nw.nodes[name] = n
	nw.sim.Record("net join %s", name)
	return n
}

// Node returns a joined endpoint by name, or nil if there is none.
func (nw *Network) Node(name string) *Node { return nw.nodes[name] }

// Nodes returns every joined node name, sorted.
//
// Sorted for the same reason sim.ActorNames is: callers iterate this to pick a
// peer to message or a victim to partition, and an order sourced from Go's map
// randomisation would make that choice depend on the runtime rather than on the
// seed. That is a determinism bug that reproduces nine times out of ten, which
// is the worst frequency for a bug to have.
func (nw *Network) Nodes() []string {
	out := make([]string, 0, len(nw.nodes))
	for name := range nw.nodes {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// Stats returns a snapshot of the counters.
func (nw *Network) Stats() Stats { return nw.stats }

// Send queues a message from one node to another. It never blocks.
//
// The message's entire fate is decided here, synchronously, while the caller
// still holds the baton: whether it is dropped, whether it is duplicated, and
// how long each surviving copy spends on the wire. Deciding at send time rather
// than inside the courier means the PRNG is consumed in exactly the order the
// program sends, which is far easier to reason about — and to debug — than an
// order that depends on how the scheduler happened to interleave couriers.
//
// Unknown node names panic. A silent no-op would turn a typo in a node name into
// a system that mysteriously never converges, and finding that by reading a
// trace is a bad afternoon.
func (nw *Network) Send(from, to string, body any) {
	nw.mustNode(from, "Send from")
	nw.mustNode(to, "Send to")

	s := nw.sim
	nw.seq++
	seq := nw.seq
	nw.stats.Sent++
	now := s.Clock().Now()

	switch {
	case nw.closed:
		nw.dropped(from, to, seq, "closed")
		return
	case nw.blocked[link{from, to}]:
		// Checked at send time, not at delivery. A message already on the wire
		// when Partition is called still arrives, which is the honest model of a
		// cut that appears between two hosts: it stops new traffic, it does not
		// reach into the wire and delete what is already travelling. A test that
		// wants the strict version partitions and then advances virtual time
		// past MaxLatency before asserting.
		nw.dropped(from, to, seq, "partition")
		return
	case s.Rand().Bool(nw.cfg.DropRate):
		nw.dropped(from, to, seq, "loss")
		return
	}

	copies := 1
	if s.Rand().Bool(nw.cfg.DuplicateRate) {
		copies = 2
		nw.stats.Duplicated++
		s.Record("net dup %s->%s #%d", from, to, seq)
	}

	for c := 1; c <= copies; c++ {
		d := nw.drawLatency()
		nw.stats.DelayedTotal += d
		nw.wire(Message{
			From:   from,
			To:     to,
			Body:   body,
			SentAt: now,
			Seq:    seq,
			Copy:   c,
		}, d)
	}
}

// drawLatency picks this copy's time on the wire.
//
// Rand.Duration consumes no randomness when the bounds are equal, so a network
// configured with a fixed latency (or none) leaves the PRNG stream untouched
// here. That is a feature: turning latency injection off does not shift every
// later draw and change an unrelated part of the run.
func (nw *Network) drawLatency() time.Duration {
	return time.Duration(nw.sim.Rand().Duration(int64(nw.cfg.MinLatency), int64(nw.cfg.MaxLatency)))
}

// wire launches the courier that carries one copy.
//
// The actor name encodes sender, receiver, message and copy number, because
// these names surface in sim's deadlock report. "net:n1->n2#41.2 waiting on
// send" identifies a stuck delivery immediately; "courier-1173" does not.
func (nw *Network) wire(m Message, d time.Duration) {
	s := nw.sim
	s.Record("net wire %s->%s #%d.%d delay=%s", m.From, m.To, m.Seq, m.Copy, d)
	s.Go(fmt.Sprintf("net:%s->%s#%d.%d", m.From, m.To, m.Seq, m.Copy), func() {
		s.Clock().Sleep(d)
		nw.deliver(m)
	})
}

// deliver runs in the courier once the wire time has elapsed.
//
// Every check here is re-evaluated now rather than at send time, because the
// interesting cases are exactly the ones where the world changed mid-flight: the
// destination crashed, the receiver stopped reading, the network shut down. A
// message sent to a healthy node and arriving at a corpse is the scenario that
// breaks retry logic, and it only exists if the crash check happens here.
func (nw *Network) deliver(m Message) {
	s := nw.sim
	switch {
	case nw.closed:
		nw.dropOnArrival(m, "closed")
	case s.Crashed(m.To):
		// Not merely pointless but actively harmful to deliver: a crashed actor
		// parked in Recv is still queued on its inbox, and handing it a value
		// would consume the message into a receiver that will never run again.
		// The message would not be "not delivered", it would be swallowed.
		nw.dropOnArrival(m, "crashed")
	case nw.nodes[m.To].inbox.Len() >= nw.cfg.InboxCapacity:
		// A full buffer implies no parked receiver — sim.Chan.Recv drains the
		// buffer before it parks — so an inbox at capacity means a Send here
		// would block this courier indefinitely. Dropping instead keeps a slow
		// reader from being reported as a network deadlock.
		nw.dropOnArrival(m, "inbox-full")
	default:
		nw.stats.Delivered++
		s.Record("net deliver %s->%s #%d.%d", m.From, m.To, m.Seq, m.Copy)
		nw.nodes[m.To].inbox.Send(m)
	}
}

func (nw *Network) dropped(from, to string, seq uint64, reason string) {
	nw.stats.Dropped++
	nw.sim.Record("net drop %s->%s #%d reason=%s", from, to, seq, reason)
}

func (nw *Network) dropOnArrival(m Message, reason string) {
	nw.stats.Dropped++
	nw.sim.Record("net drop %s->%s #%d.%d reason=%s", m.From, m.To, m.Seq, m.Copy, reason)
}

// Close shuts the network down: every inbox is closed, so nodes blocked in Recv
// wake with ok=false, and any message still on the wire is discarded on arrival.
//
// It exists so a test can end. A node's message loop is naturally an infinite
// Recv loop, and without a way to close the network the only ways to stop one
// are a counted loop (which is not how real nodes are written) or a deadlock
// report (which is not a passing test). Close is idempotent, because a shutdown
// path that panics when run twice is a shutdown path nobody trusts.
func (nw *Network) Close() {
	if nw.closed {
		return
	}
	nw.closed = true
	// Sorted, not map order. Closing wakes parked receivers, and although the
	// resulting runnable set is the same either way today, "it happens not to
	// matter here" is not a rule that survives the next edit. Sort the keys.
	for _, name := range nw.Nodes() {
		nw.nodes[name].inbox.Close()
	}
	nw.sim.Record("net close")
}

func (nw *Network) mustNode(name, what string) *Node {
	n, ok := nw.nodes[name]
	if !ok {
		panic(fmt.Sprintf("lockstep/netsim: %s: no node named %q; joined nodes are %v", what, name, nw.Nodes()))
	}
	return n
}
