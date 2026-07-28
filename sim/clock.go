package sim

import (
	"fmt"
	"sort"
	"time"
)

// Clock is virtual time.
//
// # Why not the real clock
//
// A test that waits for a five-second timeout takes five seconds, so nobody
// writes it, so timeout bugs ship. Worse, a real clock makes a test's outcome
// depend on how loaded the machine was — the definition of flaky.
//
// Virtual time fixes both. The clock is a counter that moves only when the
// simulation says so, and it moves in jumps: when every actor is blocked, it
// leaps straight to the next scheduled wakeup. A simulated day costs
// microseconds, and the result is identical on a laptop and on a loaded CI box.
//
// The payoff that matters most is control. A real test cannot arrange for a
// message to arrive one nanosecond before a deadline. A simulation can do it on
// purpose, every run, and can do the one-nanosecond-after case in the same
// breath. Timeout and retry bugs live exactly in that gap.
//
// # The rule
//
// Inside a simulation, time.Now, time.Since, time.Sleep, time.After and
// time.NewTimer are all forbidden. Every one of them reads the host clock and
// breaks reproducibility. Take a *Clock and use it, which also makes the
// dependency visible in your function signatures rather than hidden in a call.
type Clock struct {
	sim *Sim
	now time.Time
	// timers is the pending wakeup set, kept sorted by deadline then by a
	// monotonically increasing sequence number.
	//
	// The sequence tiebreak is not cosmetic. Two timers set for the same
	// instant must fire in a defined order, and "whichever the map yielded
	// first" is not one. Ordering by creation makes the schedule a function of
	// the program, which is what replay depends on.
	timers []*timer
	nextID uint64
}

type timer struct {
	id    uint64
	at    time.Time
	actor *actor
}

func newClock(s *Sim, start time.Time) *Clock {
	return &Clock{sim: s, now: start}
}

// Now returns the current virtual time.
func (c *Clock) Now() time.Time {
	c.sim.mu.Lock()
	defer c.sim.mu.Unlock()
	return c.now
}

func (c *Clock) nowLocked() time.Time { return c.now }

// Since returns the virtual time elapsed since t.
func (c *Clock) Since(t time.Time) time.Duration { return c.Now().Sub(t) }

// Sleep blocks the calling actor for a virtual duration.
//
// The actor yields, so other actors run while it sleeps — and because time only
// advances when everyone is blocked, a Sleep costs no real time at all. A
// non-positive duration is a plain Yield: it still gives other actors a chance
// to run, which is usually what a caller wants and is cheaper to reason about
// than a special case that does nothing.
func (c *Clock) Sleep(d time.Duration) {
	s := c.sim
	s.mu.Lock()
	a := s.current()
	if d <= 0 {
		if s.trace != nil {
			s.trace.add(Event{Step: s.steps, Time: c.now, Actor: a.name, Kind: EventYield})
		}
		a.yield()
		s.mu.Unlock()
		return
	}
	wake := c.now.Add(d)
	c.nextID++
	c.timers = append(c.timers, &timer{id: c.nextID, at: wake, actor: a})
	a.sleeping = true
	a.wakeAt = wake
	if s.trace != nil {
		s.trace.add(Event{Step: s.steps, Time: c.now, Actor: a.name, Kind: EventSleep, Detail: d.String()})
	}
	a.yield()
	s.mu.Unlock()
}

// advanceClockLocked moves time to the next pending wakeup and wakes every
// actor due at that instant. It reports whether it did anything.
//
// All timers at the same deadline fire together rather than one per call. That
// keeps the clock's behaviour honest — two things scheduled for the same instant
// really do become runnable at the same instant, and the scheduler then decides
// their order, which is where the interesting interleavings come from.
func (s *Sim) advanceClockLocked() bool {
	c := s.clock
	// The loop exists because of a bug worth remembering. The first version
	// advanced to the earliest deadline, discarded any timers belonging to
	// crashed actors, and returned whether it had woken anyone. If the earliest
	// timer belonged to a crashed actor and no live actor was due at that same
	// instant, it returned false — and Run takes false to mean "time cannot move
	// and nobody can run", so it reported a deadlock while later timers were
	// still queued.
	//
	// Crashing a node that is asleep on a heartbeat timer is the single most
	// common thing a crash test does, so the bug fired constantly and blamed the
	// system under test for a fault in the simulator. Discarding a dead timer is
	// progress, not the end of the world: keep advancing until a live actor
	// wakes or the queue is genuinely empty.
	for {
		if len(c.timers) == 0 {
			return false
		}
		sort.Slice(c.timers, func(i, j int) bool {
			if !c.timers[i].at.Equal(c.timers[j].at) {
				return c.timers[i].at.Before(c.timers[j].at)
			}
			return c.timers[i].id < c.timers[j].id
		})

		next := c.timers[0].at
		if next.After(c.now) {
			c.now = next
			if s.trace != nil {
				s.trace.add(Event{Step: s.steps, Time: c.now, Kind: EventClockAdvance})
			}
		}

		var remaining []*timer
		woke := false
		for _, t := range c.timers {
			if t.at.After(c.now) {
				remaining = append(remaining, t)
				continue
			}
			if t.actor.crashed {
				// Discarded rather than fired. Waking a crashed actor would
				// resurrect a process that is supposed to be dead, which
				// silently invalidates every crash test built on it.
				continue
			}
			t.actor.sleeping = false
			t.actor.wakeAt = time.Time{}
			woke = true
		}
		c.timers = remaining
		if woke {
			return true
		}
		// Everything due at this instant belonged to crashed actors. Go round
		// again and try the next deadline.
	}
}

// After returns a channel that receives once the virtual duration elapses.
//
// It is the simulation's answer to time.After, and it exists mainly so timeout
// code reads naturally in a Select. The returned channel is a sim Chan, not a Go
// channel, because a Go channel would block the actor in a way the scheduler
// cannot see and the simulation would deadlock.
func (c *Clock) After(d time.Duration) *Chan {
	ch := NewChan(c.sim, 1)
	name := fmt.Sprintf("timer-%d", c.nextTimerName())
	c.sim.Go(name, func() {
		c.Sleep(d)
		ch.Send(c.Now())
	})
	return ch
}

func (c *Clock) nextTimerName() uint64 {
	c.sim.mu.Lock()
	defer c.sim.mu.Unlock()
	c.nextID++
	return c.nextID
}
