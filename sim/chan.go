package sim

import (
	"fmt"
	"sort"
)

// Chan is a channel the scheduler understands.
//
// # Why a Go channel will not do
//
// When an actor blocks on a native Go channel, it parks inside the Go runtime.
// lockstep's scheduler has no idea that happened: as far as it is concerned the
// actor still holds the baton, so it waits for a hand-back that never comes and
// the simulation hangs.
//
// Even if that were solved, a native channel would reintroduce exactly the
// nondeterminism the simulator exists to remove. Go's select picks uniformly at
// random among ready cases, using the runtime's own generator — not the seed —
// so a program with two ready channels would take a different branch on replay.
//
// Chan therefore reimplements the parts of channel semantics a simulation needs,
// with every blocking point routed through the scheduler and every choice
// derived from the seed.
//
// # Semantics
//
// A Chan with capacity 0 is unbuffered: a send blocks until a receiver takes the
// value. With capacity n > 0, sends succeed while fewer than n values are
// buffered. Receives block while empty. Closing wakes every waiter, and a
// receive from a closed empty channel returns the zero value with ok=false —
// all matching Go, so the mental model transfers.
//
// The values are `any`. Generics would be nicer at the call site, but a
// simulation's network carries heterogeneous message types between nodes and a
// typed channel per pair is more ceremony than it is worth. Callers type-assert,
// which in practice is one line in a message loop.
type Chan struct {
	sim    *Sim
	cap    int
	buf    []any
	closed bool

	// senders and receivers are actors parked on this channel, held in FIFO
	// order.
	//
	// FIFO rather than the seeded PRNG, deliberately, and the distinction
	// matters. Go's channels have no ordering guarantee among waiters, so
	// picking randomly would be a faithful model. But a simulator must be able
	// to explore an unfair schedule *on purpose*, and it cannot do that if the
	// unfairness is baked into the primitive: a randomly-served queue makes
	// starvation a rare accident rather than something a test can force. FIFO
	// here keeps the primitive predictable, and the scheduler's choice of which
	// actor runs next remains the single place unfairness is introduced.
	senders   []*waiter
	receivers []*waiter
}

type waiter struct {
	actor *actor
	val   any
	ok    bool
	done  bool
	// from identifies which channel satisfied this waiter, for Select. It is
	// -1 for a plain Send or Recv, which have only one channel and no need to
	// ask. Without it a select woken by a direct sender hand-off knows the value
	// but not its origin, and a caller switching on the index would dispatch the
	// message to the wrong handler — a bug that looks like message corruption
	// and is nothing of the sort.
	from int
}

// NewChan creates a channel with the given buffer capacity.
func NewChan(s *Sim, capacity int) *Chan {
	if capacity < 0 {
		panic("lockstep: NewChan capacity must not be negative")
	}
	return &Chan{sim: s, cap: capacity}
}

// Send delivers a value, blocking if the channel is full or unbuffered with no
// waiting receiver. Sending on a closed channel panics, matching Go.
func (c *Chan) Send(v any) {
	s := c.sim
	s.mu.Lock()
	defer s.mu.Unlock()
	a := s.current()

	if c.closed {
		panic("lockstep: send on closed Chan")
	}

	// A parked receiver takes the value directly. This is the rendezvous case
	// and it must be handled before the buffer, or an unbuffered channel could
	// never make progress.
	if len(c.receivers) > 0 {
		w := c.receivers[0]
		c.receivers = c.receivers[1:]
		w.val, w.ok, w.done = v, true, true
		c.unblock(w.actor)
		if s.trace != nil {
			s.trace.add(Event{Step: s.steps, Time: s.clock.nowLocked(), Actor: a.name, Kind: EventSend})
		}
		return
	}

	if len(c.buf) < c.cap {
		c.buf = append(c.buf, v)
		if s.trace != nil {
			s.trace.add(Event{Step: s.steps, Time: s.clock.nowLocked(), Actor: a.name, Kind: EventSend})
		}
		return
	}

	// Block until a receiver arrives.
	w := &waiter{actor: a, val: v, from: -1}
	c.senders = append(c.senders, w)
	a.blockedOn = "send"
	if s.trace != nil {
		s.trace.add(Event{Step: s.steps, Time: s.clock.nowLocked(), Actor: a.name, Kind: EventBlockSend})
	}
	a.yield()
	if c.closed && !w.done {
		panic("lockstep: send on closed Chan")
	}
}

// Recv takes a value, blocking while empty. ok is false if the channel is closed
// and drained.
func (c *Chan) Recv() (v any, ok bool) {
	s := c.sim
	s.mu.Lock()
	defer s.mu.Unlock()
	a := s.current()

	if len(c.buf) > 0 {
		v = c.buf[0]
		c.buf = c.buf[1:]
		// Taking from the buffer frees a slot, so a blocked sender can now
		// deposit. Doing this immediately keeps the buffer full whenever senders
		// are waiting, which is what a real channel does.
		if len(c.senders) > 0 {
			w := c.senders[0]
			c.senders = c.senders[1:]
			c.buf = append(c.buf, w.val)
			w.done = true
			c.unblock(w.actor)
		}
		if s.trace != nil {
			s.trace.add(Event{Step: s.steps, Time: s.clock.nowLocked(), Actor: a.name, Kind: EventRecv})
		}
		return v, true
	}

	// Unbuffered rendezvous: take straight from a parked sender.
	if len(c.senders) > 0 {
		w := c.senders[0]
		c.senders = c.senders[1:]
		w.done = true
		c.unblock(w.actor)
		if s.trace != nil {
			s.trace.add(Event{Step: s.steps, Time: s.clock.nowLocked(), Actor: a.name, Kind: EventRecv})
		}
		return w.val, true
	}

	if c.closed {
		return nil, false
	}

	w := &waiter{actor: a, from: -1}
	c.receivers = append(c.receivers, w)
	a.blockedOn = c.blockLabel()
	if s.trace != nil {
		s.trace.add(Event{Step: s.steps, Time: s.clock.nowLocked(), Actor: a.name, Kind: EventBlockRecv})
	}
	a.yield()
	return w.val, w.ok
}

// TryRecv takes a value if one is immediately available, without blocking.
//
// Provided because a poll loop written with Recv plus a timeout is both slower
// and harder to reason about, and because "is there anything for me right now"
// is a normal question for a node's main loop to ask.
func (c *Chan) TryRecv() (v any, ok bool) {
	s := c.sim
	s.mu.Lock()
	defer s.mu.Unlock()
	s.current()

	if len(c.buf) > 0 {
		v = c.buf[0]
		c.buf = c.buf[1:]
		if len(c.senders) > 0 {
			w := c.senders[0]
			c.senders = c.senders[1:]
			c.buf = append(c.buf, w.val)
			w.done = true
			c.unblock(w.actor)
		}
		return v, true
	}
	if len(c.senders) > 0 {
		w := c.senders[0]
		c.senders = c.senders[1:]
		w.done = true
		c.unblock(w.actor)
		return w.val, true
	}
	return nil, false
}

// Len returns the number of buffered values.
func (c *Chan) Len() int {
	c.sim.mu.Lock()
	defer c.sim.mu.Unlock()
	return len(c.buf)
}

// Close closes the channel and wakes every waiter. Closing twice panics,
// matching Go.
func (c *Chan) Close() {
	s := c.sim
	s.mu.Lock()
	defer s.mu.Unlock()
	if c.closed {
		panic("lockstep: close of closed Chan")
	}
	c.closed = true
	for _, w := range c.receivers {
		w.val, w.ok, w.done = nil, false, true
		c.unblock(w.actor)
	}
	c.receivers = nil
	for _, w := range c.senders {
		c.unblock(w.actor)
	}
	c.senders = nil
}

func (c *Chan) unblock(a *actor) {
	a.blockedOn = ""
}

func (c *Chan) blockLabel() string { return "recv" }

// Select waits until one of the given channels can receive, then returns its
// index and value.
//
// # The one place randomness is correct
//
// When several channels are ready simultaneously, Select consults the seeded
// PRNG to choose among them. That is deliberate and it is the opposite of the
// FIFO policy inside Chan: here, a fixed preference would mean a program that
// always drained its first channel would never exercise the second, and a whole
// class of ordering bugs would be invisible. Randomising the choice — from the
// seed, so it still replays — is what makes those orderings reachable.
//
// The candidate set is built by index and sorted before choosing, so the PRNG's
// output always maps to the same channel across runs.
func Select(s *Sim, chans ...*Chan) (index int, val any, ok bool) {
	for {
		s.mu.Lock()
		a := s.current()

		var ready []int
		for i, c := range chans {
			if len(c.buf) > 0 || len(c.senders) > 0 || c.closed {
				ready = append(ready, i)
			}
		}
		sort.Ints(ready)

		if len(ready) > 0 {
			pick := ready[s.rand.Intn(len(ready))]
			s.mu.Unlock()
			v, o := chans[pick].Recv()
			return pick, v, o
		}

		// Nothing ready. Park on every channel at once, then let the scheduler
		// decide. Registering on all of them is what makes this a real select
		// rather than a poll: whichever channel becomes ready first wakes this
		// actor.
		//
		// One waiter PER channel, each tagged with its index. A single shared
		// waiter would be woken correctly but would not record which channel
		// delivered, and Select's whole contract is returning that index.
		ws := make([]*waiter, len(chans))
		for i, c := range chans {
			w := &waiter{actor: a, from: i}
			ws[i] = w
			c.receivers = append(c.receivers, w)
		}
		a.blockedOn = fmt.Sprintf("select(%d)", len(chans))
		if s.trace != nil {
			s.trace.add(Event{Step: s.steps, Time: s.clock.nowLocked(), Actor: a.name, Kind: EventBlockRecv, Detail: "select"})
		}
		a.yield()

		// Woken. Deregister from every channel before doing anything else, or
		// the stale entries accumulate and a later send hands a value to an
		// actor that is no longer listening — a message that vanishes, surfacing
		// much later as an unexplained timeout.
		fired := -1
		var fw *waiter
		for i, c := range chans {
			out := c.receivers[:0]
			for _, r := range c.receivers {
				if r != ws[i] {
					out = append(out, r)
				}
			}
			c.receivers = out
			if ws[i].done {
				// At most one can have fired: a sender removes the waiter from
				// its own queue before marking it done, and the actor was
				// blocked throughout, so no second sender could have run.
				fired, fw = i, ws[i]
			}
		}

		if fired >= 0 {
			s.mu.Unlock()
			return fired, fw.val, fw.ok
		}
		// Woken without a direct hand-off — a channel was closed or buffered.
		// Loop and re-scan.
		s.mu.Unlock()
	}
}
