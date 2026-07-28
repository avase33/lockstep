// Package sim is the deterministic core of lockstep: a scheduler, a virtual
// clock, and a single seeded source of randomness.
//
// # The one guarantee
//
// Given the same seed and the same code, a simulation produces the same
// sequence of events, every time, on every machine. That is the whole product.
// A bug found on someone's laptop reproduces byte for byte in CI, and a fix can
// be proven rather than hoped for.
//
// Everything in this package exists to protect that guarantee, and the design is
// shaped by one uncomfortable fact: Go's runtime scheduler is deliberately
// nondeterministic. Goroutines are preempted at points the runtime chooses,
// select picks uniformly at random among ready cases, and map iteration order is
// randomised on purpose. A simulation that lets any of that leak in will
// reproduce most of the time, which is worse than never — it produces trust that
// is occasionally betrayed.
//
// # How determinism is achieved
//
// Simulated actors still run as ordinary goroutines, because that is what makes
// lockstep usable: you write normal sequential Go, not a state machine chopped
// into callbacks. But at most ONE actor goroutine is runnable at any instant.
// The scheduler holds a baton. An actor runs until it performs an operation that
// yields — sleeping, sending, receiving, or an explicit Yield — at which point
// it hands the baton back and blocks. The scheduler then picks the next actor to
// run, using only the seeded PRNG to break ties.
//
// So the Go scheduler is never asked to choose between two runnable actor
// goroutines, because there is never more than one. Its nondeterminism is
// removed from the equation rather than fought.
//
// The cost is real and worth stating: a lockstep simulation has no parallelism,
// so it cannot find bugs that require two instructions to genuinely execute at
// the same instant on two cores. It finds concurrency bugs — orderings,
// interleavings, races between logical steps — not memory-model bugs. Use the
// race detector for the latter. This is the same trade FoundationDB and
// TigerBeetle make, and it is the reason their simulations are worth running.
//
// # Virtual time
//
// The clock is a counter, not a reading of the host clock. Time advances only
// when every actor is blocked, and then it jumps directly to the next scheduled
// wakeup. A simulated hour costs microseconds, and a test that would take a day
// of real waiting runs in a second.
//
// This is also why a timeout bug is findable here and nearly impossible to find
// live: the simulation can place a message delivery exactly one nanosecond
// before and exactly one nanosecond after a deadline, deliberately, on demand.
//
// # What will break determinism
//
// These are not hypothetical. Each one has cost someone a week:
//
//   - time.Now, time.Sleep, time.After — use the Clock.
//   - math/rand's global functions, crypto/rand — use Rand.
//   - Iterating a map and acting on the order. Sort the keys first. lockstep's
//     own code does this everywhere, and TestSameSeedProducesIdenticalTrace
//     exists to catch it when someone forgets.
//   - Real goroutines that are not registered actors.
//   - Real network or disk I/O.
//   - sync.WaitGroup, unbuffered channels between actors, or any other blocking
//     primitive the scheduler does not know about — the scheduler cannot hand
//     the baton to a goroutine it does not know is waiting, so the simulation
//     deadlocks. Use the Chan type in this package.
//
// Run returns a Result whose Error reports a deadlock rather than hanging, and
// the message names the actors it was still waiting for — the alternative being
// a test that times out in CI with no explanation.
package sim

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// Seed identifies a simulation run. Printing it is what makes a failure
// reproducible, so it is a named type rather than a bare uint64 — it shows up in
// output as a hex literal that can be pasted straight back into a flag.
type Seed uint64

func (s Seed) String() string { return fmt.Sprintf("0x%08x", uint64(s)) }

// Sim is a running simulation. It is created by New and driven by Run.
//
// Sim is not safe for concurrent use from outside the simulation, and that is
// deliberate: calling into a Sim from a goroutine the scheduler does not manage
// is the single most common way to break determinism, so the API gives no
// reason to.
type Sim struct {
	seed  Seed
	rand  *Rand
	clock *Clock

	mu      sync.Mutex
	cond    *sync.Cond
	running int // index into actors of the actor holding the baton, or -1
	actors  []*actor
	byName  map[string]*actor

	// blocked counts actors waiting on something the scheduler will wake:
	// a timer, a channel, or another actor. When every live actor is blocked and
	// no timer is pending, the simulation is finished or deadlocked.
	trace   *Trace
	failure error
	done    bool
	steps   int
	maxStep int
}

// Config controls a simulation run.
type Config struct {
	// Seed makes the run reproducible. A zero Seed means "pick one and print
	// it", which is what you want in CI: every run explores a new schedule, and
	// a failure names the seed that found it.
	Seed Seed

	// MaxSteps bounds the number of scheduling decisions before the run is
	// declared stuck. It exists because a livelock — two actors politely
	// yielding to each other forever — is otherwise indistinguishable from slow
	// progress. Default DefaultMaxSteps.
	MaxSteps int

	// DisableTrace turns off event recording.
	//
	// Phrased negatively so the zero value keeps tracing ON. That is not a style
	// preference: an earlier version had `Trace bool` documented as defaulting to
	// true, which Go zero-valued to false, so every Config that did not mention
	// it silently recorded nothing. The determinism test then compared two empty
	// traces, found them equal, and passed — a green test asserting nothing.
	// A field whose safe state is its zero value cannot fail that way.
	//
	// Recording costs one append per scheduling decision. Disable it only for
	// benchmarks.
	DisableTrace bool

	// StartTime is the virtual clock's initial reading. Defaults to a fixed
	// instant rather than the host's current time, because a simulation whose
	// starting point moves is not reproducible tomorrow.
	StartTime time.Time
}

// DefaultMaxSteps bounds a run at a million scheduling decisions.
const DefaultMaxSteps = 1_000_000

// DefaultStartTime is the virtual clock's default origin. A fixed, obviously
// artificial instant, so a timestamp appearing in output is recognisably
// simulated rather than mistaken for a real one.
var DefaultStartTime = time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)

func (c Config) withDefaults() Config {
	if c.MaxSteps <= 0 {
		c.MaxSteps = DefaultMaxSteps
	}
	if c.StartTime.IsZero() {
		c.StartTime = DefaultStartTime
	}
	return c
}

// actor is one simulated goroutine.
type actor struct {
	id       int
	name     string
	sim      *Sim
	resume   chan struct{} // scheduler signals here to hand over the baton
	finished bool
	crashed  bool
	// waking is the virtual time this actor should next become runnable, or
	// zero if it is not sleeping.
	wakeAt   time.Time
	sleeping bool
	// blockedOn names what the actor is waiting for, purely so a deadlock
	// report can say something useful instead of "everything is stuck".
	blockedOn string
	panicVal  any
	panicStk  []byte
}

// New creates a simulation. Actors are added with Go; the run starts at Run.
func New(cfg Config) *Sim {
	cfg = cfg.withDefaults()
	s := &Sim{
		seed:    cfg.Seed,
		rand:    NewRand(uint64(cfg.Seed)),
		running: -1,
		byName:  make(map[string]*actor),
		maxStep: cfg.MaxSteps,
	}
	s.cond = sync.NewCond(&s.mu)
	s.clock = newClock(s, cfg.StartTime)
	if !cfg.DisableTrace {
		s.trace = &Trace{}
	}
	return s
}

// Seed returns the seed this run was created with.
func (s *Sim) Seed() Seed { return s.seed }

// Rand returns the simulation's only source of randomness.
//
// Every non-deterministic choice a test makes must come from here — which key to
// write, how long to wait, whether to crash a node. Using math/rand instead
// produces a simulation that finds bugs it cannot reproduce, which wastes more
// time than it saves.
func (s *Sim) Rand() *Rand { return s.rand }

// Clock returns the virtual clock. See the Clock type for why time.Now is
// forbidden.
func (s *Sim) Clock() *Clock { return s.clock }

// Trace returns the recorded event trace, or nil if tracing was disabled.
func (s *Sim) Trace() *Trace { return s.trace }

// Go registers an actor and starts its goroutine, which immediately blocks
// waiting for the baton.
//
// Names must be unique and are used in traces and deadlock reports. A name like
// "node-3" or "client-1" makes a failure readable; "goroutine" does not.
//
// Go may be called before Run or from inside a running actor — spawning a
// request handler or a timer mid-simulation is ordinary and supported.
// Determinism survives because the spawn happens at a fixed point in the
// schedule: the parent actor holds the baton when it calls Go, so the new actor
// joins the runnable set at a position that is itself a function of the seed.
//
// What is NOT supported is calling Go from a goroutine the scheduler does not
// manage. That inserts an actor at a moment determined by the Go runtime rather
// than by the seed, and replay stops working.
func (s *Sim) Go(name string, fn func()) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done {
		panic("lockstep: Go called after the simulation finished")
	}
	if _, dup := s.byName[name]; dup {
		panic(fmt.Sprintf("lockstep: duplicate actor name %q; names must be unique so traces are readable", name))
	}
	a := &actor{
		id:     len(s.actors),
		name:   name,
		sim:    s,
		resume: make(chan struct{}),
	}
	s.actors = append(s.actors, a)
	s.byName[name] = a

	go func() {
		// Wait for the scheduler's first hand-off. Nothing in fn runs until the
		// scheduler decides this actor goes first.
		<-a.resume
		defer func() {
			if r := recover(); r != nil {
				a.panicVal = r
				a.panicStk = stack()
			}
			s.mu.Lock()
			a.finished = true
			s.running = -1
			s.cond.Broadcast()
			s.mu.Unlock()
		}()
		fn()
	}()
}

// Run executes the simulation until every actor finishes, the step budget is
// exhausted, or the simulation deadlocks.
//
// It returns a Result rather than an error so a caller can inspect the trace and
// the finishing state even on success. Call Result.Check in a test to turn a
// failure into a t.Fatal with a reproducible seed.
func (s *Sim) Run() *Result {
	s.mu.Lock()
	defer s.mu.Unlock()

	for {
		if s.steps >= s.maxStep {
			s.failure = &StuckError{
				Steps: s.steps,
				Seed:  s.seed,
				Note:  "step budget exhausted; the simulation is making decisions but not finishing (livelock, or MaxSteps is too low)",
			}
			break
		}

		runnable := s.runnableLocked()
		if len(runnable) == 0 {
			// Nobody can run right now. Either time can move forward and wake a
			// sleeper, or nothing can ever happen again.
			if s.advanceClockLocked() {
				continue
			}
			if s.allFinishedLocked() {
				break
			}
			s.failure = s.deadlockErrorLocked()
			break
		}

		// The scheduling decision. Every non-determinism in the whole system
		// funnels through this one line, which is why it is the only place the
		// PRNG is consulted for scheduling.
		pick := runnable[s.rand.Intn(len(runnable))]
		s.steps++
		if s.trace != nil {
			s.trace.add(Event{
				Step:  s.steps,
				Time:  s.clock.nowLocked(),
				Actor: pick.name,
				Kind:  EventSchedule,
			})
		}
		s.handOffLocked(pick)

		if err := s.collectPanicLocked(); err != nil {
			s.failure = err
			break
		}
	}

	s.done = true
	res := &Result{
		Seed:  s.seed,
		Steps: s.steps,
		Err:   s.failure,
		trace: s.trace,
		end:   s.clock.nowLocked(),
	}
	return res
}

// runnableLocked returns the actors that could run right now, in a
// deterministic order.
//
// The sort is not decoration. The candidate set is built by walking s.actors,
// which is already ordered, but callers elsewhere build sets from maps; sorting
// by actor id here means the selection index the PRNG produces always refers to
// the same actor across runs. Without it, the PRNG would be deterministic and
// the simulation still would not be.
func (s *Sim) runnableLocked() []*actor {
	var out []*actor
	for _, a := range s.actors {
		if a.finished || a.crashed || a.sleeping || a.blockedOn != "" {
			continue
		}
		out = append(out, a)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].id < out[j].id })
	return out
}

func (s *Sim) allFinishedLocked() bool {
	for _, a := range s.actors {
		if !a.finished && !a.crashed {
			return false
		}
	}
	return true
}

// handOffLocked gives the baton to one actor and waits for it to give it back.
//
// This is the heart of the scheduler and the reason at most one actor is ever
// runnable: control does not return here until the chosen actor blocks or
// finishes, so no second actor is ever released concurrently.
func (s *Sim) handOffLocked(a *actor) {
	s.running = a.id
	s.mu.Unlock()
	a.resume <- struct{}{}
	s.mu.Lock()
	for s.running == a.id && !a.finished {
		s.cond.Wait()
	}
}

// yield is called from inside an actor to give the baton back and block until
// the scheduler picks it again.
func (a *actor) yield() {
	s := a.sim
	s.running = -1
	s.cond.Broadcast()
	s.mu.Unlock()
	<-a.resume
	s.mu.Lock()
}

func (s *Sim) collectPanicLocked() error {
	for _, a := range s.actors {
		if a.panicVal != nil {
			return &PanicError{
				Actor: a.name,
				Value: a.panicVal,
				Stack: string(a.panicStk),
				Seed:  s.seed,
			}
		}
	}
	return nil
}

// deadlockErrorLocked builds a report naming who is stuck and on what. The
// detail matters: "deadlock" alone sends someone hunting through a trace, while
// "node-2 waiting on recv(inbox), node-3 waiting on recv(inbox), no timers
// pending" usually identifies the bug on sight.
func (s *Sim) deadlockErrorLocked() error {
	var waiting []string
	for _, a := range s.actors {
		if a.finished || a.crashed {
			continue
		}
		what := a.blockedOn
		if a.sleeping {
			what = "sleep"
		}
		if what == "" {
			what = "unknown"
		}
		waiting = append(waiting, fmt.Sprintf("%s waiting on %s", a.name, what))
	}
	sort.Strings(waiting)
	return &DeadlockError{
		Seed:    s.seed,
		Steps:   s.steps,
		Waiting: waiting,
		At:      s.clock.nowLocked(),
	}
}

// current returns the actor holding the baton, panicking with a useful message
// if called from an unmanaged goroutine.
//
// The panic is the point. Calling a sim primitive from a raw goroutine is a
// determinism bug, and failing loudly at the call site is enormously cheaper to
// diagnose than a simulation that quietly stops reproducing.
func (s *Sim) current() *actor {
	if s.running < 0 || s.running >= len(s.actors) {
		panic("lockstep: simulation primitive called from outside a registered actor; " +
			"every goroutine in a simulation must be started with Sim.Go")
	}
	return s.actors[s.running]
}

// Yield hands control back to the scheduler without blocking.
//
// Use it to expose an interleaving the scheduler would otherwise never explore:
// placing a Yield between two statements tells lockstep that another actor may
// run at exactly that point. It is the simulation equivalent of a preemption
// point, and adding them is how a test goes from "checks the happy path" to
// "checks every ordering of these three steps".
func (s *Sim) Yield() {
	s.mu.Lock()
	a := s.current()
	if s.trace != nil {
		s.trace.add(Event{Step: s.steps, Time: s.clock.nowLocked(), Actor: a.name, Kind: EventYield})
	}
	a.yield()
	s.mu.Unlock()
}

// Crash stops an actor immediately and permanently.
//
// The actor's goroutine is left blocked forever rather than killed, because Go
// provides no way to kill a goroutine and pretending otherwise would leak
// surprises. The cost is one parked goroutine per crashed actor for the run's
// duration, which is bounded and cheap; the benefit is that a crash means
// exactly what it says — the actor stops mid-operation, holding whatever state
// it held, and never resumes.
//
// This models a process being killed, not a graceful shutdown. Modelling
// restart is the caller's job: start a fresh actor whose state is whatever the
// crashed one had durably persisted, which is precisely the question a crash
// test should be asking.
func (s *Sim) Crash(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	a, ok := s.byName[name]
	if !ok {
		panic(fmt.Sprintf("lockstep: Crash(%q): no such actor", name))
	}
	if a.finished || a.crashed {
		return
	}
	a.crashed = true
	a.blockedOn = "crashed"
	if s.trace != nil {
		s.trace.add(Event{Step: s.steps, Time: s.clock.nowLocked(), Actor: name, Kind: EventCrash})
	}
	if s.running == a.id {
		s.running = -1
		s.cond.Broadcast()
	}
}

// Crashed reports whether an actor has been crashed.
func (s *Sim) Crashed(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	a, ok := s.byName[name]
	return ok && a.crashed
}

// ActorNames returns every registered actor name, sorted.
//
// Sorted because callers iterate this to pick a victim to crash, and an
// unsorted order sourced from a map would make that choice depend on Go's map
// randomisation rather than on the seed.
func (s *Sim) ActorNames() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, 0, len(s.actors))
	for _, a := range s.actors {
		out = append(out, a.name)
	}
	sort.Strings(out)
	return out
}

// Result is the outcome of a run.
type Result struct {
	// Seed is the seed that produced this run. Print it on failure; it is the
	// entire reproduction recipe.
	Seed Seed
	// Steps is the number of scheduling decisions made.
	Steps int
	// Err is nil on a clean finish, or a *DeadlockError, *PanicError,
	// *StuckError or a user-reported failure.
	Err error

	trace *Trace
	end   time.Time
}

// Trace returns the event trace, or nil if tracing was disabled.
func (r *Result) Trace() *Trace { return r.trace }

// VirtualTime returns how much simulated time elapsed.
func (r *Result) VirtualTime() time.Duration { return r.end.Sub(DefaultStartTime) }

// Failed reports whether the run ended badly.
func (r *Result) Failed() bool { return r.Err != nil }

// Error renders the failure with its reproduction command, or "" on success.
func (r *Result) Error() string {
	if r.Err == nil {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%v\n", r.Err)
	fmt.Fprintf(&b, "\n  seed:  %s\n", r.Seed)
	fmt.Fprintf(&b, "  steps: %d\n", r.Steps)
	fmt.Fprintf(&b, "  repro: go test -run <YourTest> -lockstep.seed=%s\n", r.Seed)
	return b.String()
}

// DeadlockError means no actor could make progress and no timer was pending.
type DeadlockError struct {
	Seed    Seed
	Steps   int
	Waiting []string
	At      time.Time
}

func (e *DeadlockError) Error() string {
	return fmt.Sprintf("deadlock at virtual time %s after %d steps:\n    %s",
		e.At.Sub(DefaultStartTime), e.Steps, strings.Join(e.Waiting, "\n    "))
}

// PanicError means an actor panicked. The simulation stops at that point.
type PanicError struct {
	Actor string
	Value any
	Stack string
	Seed  Seed
}

func (e *PanicError) Error() string {
	return fmt.Sprintf("actor %s panicked: %v\n%s", e.Actor, e.Value, e.Stack)
}

// StuckError means the step budget ran out.
type StuckError struct {
	Steps int
	Seed  Seed
	Note  string
}

func (e *StuckError) Error() string {
	return fmt.Sprintf("simulation did not finish within %d steps: %s", e.Steps, e.Note)
}
