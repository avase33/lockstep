package kvstore

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/avase33/lockstep/linz"
	"github.com/avase33/lockstep/netsim"
	"github.com/avase33/lockstep/sim"
)

// Mode selects which read path the workload drives. It exists so that one
// workload definition serves both implementations: the fixed store must be
// judged by exactly the same clients, faults and seeds as the broken one, or a
// clean result proves nothing about the read path.
type Mode uint8

const (
	// SingleReplicaReads drives Store, whose Get returns the first reply from
	// any replica. This is the mode that should fail.
	SingleReplicaReads Mode = iota

	// QuorumReads drives QuorumStore, whose Get reads from a quorum and writes
	// the winner back before returning. This is the mode that should never fail,
	// and it is what makes a failure in the other mode credible.
	QuorumReads
)

func (m Mode) String() string {
	switch m {
	case SingleReplicaReads:
		return "single-replica reads"
	case QuorumReads:
		return "quorum reads"
	}
	return fmt.Sprintf("Mode(%d)", uint8(m))
}

// Options describes one simulated run. The zero value is the configuration this
// example is tuned for, so a caller who only wants to switch Mode can leave the
// rest alone.
//
// The defaults are chosen against a constraint that is easy to miss: linz's cost
// is linear in how many operations a history contains and exponential in how
// many of them overlap on a single key. Four clients over three keys keeps the
// per-key concurrency in the low single digits, where the check costs
// microseconds. Raising Clients is the one field that can turn a fast test into
// one that never finishes.
type Options struct {
	// Mode selects the read path under test.
	Mode Mode

	// Cluster configures the replicas, the timeouts and the durability delay.
	Cluster ClusterConfig

	// Faults configures the network. The zero value of netsim.Config is a
	// perfect network, so this is set by withDefaults rather than left empty:
	// a run of this example with no faults is not a demonstration of anything,
	// and defaulting to silence is how fault injection quietly switches itself
	// off.
	Faults netsim.Config

	// Clients is how many client actors run concurrently, Ops how many
	// operations each performs, and Keys how many distinct keys they share.
	Clients int
	Ops     int
	Keys    int

	// WriteRatio is the probability that an operation is a write.
	//
	// Reads have to outnumber writes for the bug to show up often — a stale read
	// needs a completed write to be stale relative to — but writes have to be
	// frequent enough that replicas actually diverge. Two reads per write is a
	// reasonable middle and is also roughly what real workloads look like.
	WriteRatio float64

	// ThinkTime bounds the pause between one client operation and the next,
	// drawn from Sim.Rand. Non-zero so clients drift out of phase with each
	// other instead of marching in lockstep, which would explore one interleaving
	// over and over.
	ThinkTime time.Duration

	// Partition, when true, isolates one replica from the cluster partway
	// through the run and heals it later.
	Partition bool

	// CrashRestart, when true, kills one replica's process partway through the
	// run and brings it back as a new incarnation.
	CrashRestart bool
}

func (o Options) withDefaults() Options {
	o.Cluster = o.Cluster.withDefaults()
	if o.Clients <= 0 {
		o.Clients = 4
	}
	if o.Ops <= 0 {
		o.Ops = 12
	}
	if o.Keys <= 0 {
		o.Keys = 3
	}
	if o.WriteRatio <= 0 {
		o.WriteRatio = 0.34
	}
	if o.ThinkTime <= 0 {
		o.ThinkTime = 8 * time.Millisecond
	}
	if o.Faults == (netsim.Config{}) {
		o.Faults = netsim.Config{
			// Wide enough that two messages sent together routinely arrive
			// backwards, which is what lets one replica be behind another at the
			// instant a read lands on it.
			MinLatency: 1 * time.Millisecond,
			MaxLatency: 20 * time.Millisecond,
			// Low rates on purpose. Heavy loss would make the store fail to reach
			// a quorum most of the time, and a history of operations that never
			// returned is a history that constrains nothing.
			DropRate:      0.04,
			DuplicateRate: 0.06,
		}
	}
	return o
}

// DefaultOptions returns the options this example is tuned for, with the given
// read path selected and every fault switched on. Use it rather than a bare
// Options literal when you want the defaults to be visible in the caller.
func DefaultOptions(mode Mode) Options {
	o := Options{Mode: mode}.withDefaults()
	o.Partition = true
	o.CrashRestart = true
	return o
}

// Normalised returns the options Run would actually use, with every zero field
// replaced by its default.
//
// It exists so that a caller printing the configuration prints what the run will
// do rather than what was typed. The two differ for every field left at zero,
// and a demo that announced "0% loss" while injecting 4% would be worse than
// saying nothing.
func (o Options) Normalised() Options { return o.withDefaults() }

// Outcome is everything one seeded run produced: what the clients observed,
// what the checker made of it, and enough about the run itself to prove two
// runs were the same run.
type Outcome struct {
	// Seed is the reproduction recipe. Print it on failure; it is the only thing
	// anyone needs.
	Seed sim.Seed

	// Mode is the read path that was under test.
	Mode Mode

	// Check is the linearizability verdict. Note that Check.Status can be
	// Unknown, which is neither a pass nor a fail; see linz.Unknown.
	Check linz.Result

	// History is the recording the verdict was reached from, kept so a
	// surprising result can be printed in full.
	History *linz.History

	// TraceHash digests every scheduling decision the simulation made. Two runs
	// of the same seed that agree on it explored the same interleaving down to
	// the last step, which is what makes a reproduction a reproduction rather
	// than a second run that happened to fail too.
	TraceHash string

	// Trace is the run's event log: who ran, when time moved, which messages
	// were delayed, dropped or delivered. Trace.Tail is what to reach for after
	// a failure — the beginning of a run is rarely where the bug is.
	Trace *sim.Trace

	// Steps is the number of scheduling decisions, VirtualTime how much
	// simulated time the run covered, and Net what the network did to the
	// messages.
	Steps       int
	VirtualTime time.Duration
	Net         netsim.Stats

	// Faults is the fault-injection log. Check it before believing a clean
	// result: a run that injected nothing proves nothing.
	Faults []string

	// SimErr is non-nil if the simulation itself failed — a deadlock, a panic in
	// an actor, or an exhausted step budget. It is a bug in this example or in
	// lockstep, never a finding about the store, and it must be treated as a
	// failure rather than folded into the checker's verdict.
	SimErr error

	// Returned counts operations whose client saw a response, and Pending those
	// that never returned.
	Returned int
	Pending  int

	// Diverged reports whether two replicas ever disagreed about a key during
	// the run. It is the precondition the single-replica read bug needs,
	// recorded so a test can assert the workload still produces it.
	Diverged bool
}

// Violated reports whether the checker proved this run wrong. It is deliberately
// not the negation of OK: an Unknown verdict is neither, and treating it as
// either is how a checker gets quietly switched off.
func (o Outcome) Violated() bool { return o.Check.Status == linz.NotLinearizable }

// Clean reports whether the run finished and was proved linearizable.
func (o Outcome) Clean() bool { return o.SimErr == nil && o.Check.Status == linz.Linearizable }

// Repro is the command that runs this exact seed again.
func (o Outcome) Repro() string {
	flag := ""
	if o.Mode == QuorumReads {
		flag = " -mode=quorum"
	}
	return fmt.Sprintf("go run ./examples/kvstore/cmd -seed=%s%s", o.Seed, flag)
}

// Digest renders everything about a run that must not vary between replays, and
// nothing that may.
//
// Comparing digests is how TestReproducibleFromSeed distinguishes "the bug
// happened again" from "the same bug, in the same place, by the same route".
// The trace hash covers the schedule and the failure report covers the verdict;
// wall-clock time and the checker's elapsed time are excluded because they
// legitimately differ, and including them would produce a determinism test that
// fails for a reason that has nothing to do with determinism.
func (o Outcome) Digest() string {
	var b strings.Builder
	fmt.Fprintf(&b, "seed=%s mode=%s trace=%s steps=%d virtual=%s\n",
		o.Seed, o.Mode, o.TraceHash, o.Steps, o.VirtualTime)
	fmt.Fprintf(&b, "net=%s\n", o.Net)
	fmt.Fprintf(&b, "faults=%s\n", strings.Join(o.Faults, "; "))
	fmt.Fprintf(&b, "status=%s operations=%d pending=%d partitions=%d widest=%d search-steps=%d\n",
		o.Check.Status, o.Check.Operations, o.Check.Pending, o.Check.Partitions, o.Check.Widest, o.Check.Steps)
	if o.Check.Failure != nil {
		b.WriteString(o.Check.Failure.String())
	}
	return b.String()
}

// Run executes one simulation of the workload against the chosen read path and
// checks what the clients observed.
//
// Everything the run does is a function of the seed: the keys chosen, the
// operation types, the think time, the network's latencies and losses, which
// replica gets isolated and when, which one is killed and for how long, and the
// order in which every actor takes its turn. That is what makes the Outcome
// worth keeping.
func Run(seed sim.Seed, opt Options) Outcome {
	opt = opt.withDefaults()

	s := sim.New(sim.Config{Seed: seed})
	net := netsim.New(s, opt.Faults)
	cluster := NewCluster(s, net, opt.Cluster)
	history := linz.NewHistory()

	// Buffered to the number of clients so a finishing client never parks: the
	// driver may not be scheduled for a while, and a client blocked on a
	// hand-off it does not care about would drag out every run for no reason.
	done := sim.NewChan(s, opt.Clients)

	for i := 0; i < opt.Clients; i++ {
		store := openStore(cluster, opt.Mode, i)
		s.Go(fmt.Sprintf("client-%d", i), func() {
			runClient(s, history, store, i, opt)
			done.Send(nil)
		})
	}

	if opt.Partition || opt.CrashRestart {
		s.Go("chaos", func() { injectFaults(cluster, opt) })
	}

	// The network stays up until every client has finished, and is then closed
	// so the replicas' Recv loops end. Without that the run would finish with
	// three actors parked forever and be reported as a deadlock — a correct
	// report of a test that forgot to end.
	s.Go("driver", func() {
		for i := 0; i < opt.Clients; i++ {
			done.Recv()
		}
		net.Close()
	})

	res := s.Run()

	out := Outcome{
		Seed:        seed,
		Mode:        opt.Mode,
		History:     history,
		TraceHash:   res.Trace().Hash(),
		Trace:       res.Trace(),
		Steps:       res.Steps,
		VirtualTime: res.VirtualTime(),
		Net:         net.Stats(),
		Faults:      cluster.Faults(),
		SimErr:      res.Err,
		Diverged:    cluster.Diverged(),
	}
	for _, op := range history.Operations() {
		if op.Pending {
			out.Pending++
		} else {
			out.Returned++
		}
	}
	out.Check = linz.Check(linz.NewKVModel(), history)
	return out
}

// openStore builds the client handle for the read path under test. Both stores
// wrap the same Client, so switching modes changes one method and nothing else.
func openStore(c *Cluster, mode Mode, id int) KV {
	client := c.Client(id)
	if mode == QuorumReads {
		return &QuorumStore{Client: client}
	}
	return &Store{Client: client}
}

// runClient is one client's whole workload: a sequence of reads and writes on
// shared keys, recorded into the history as it goes.
//
// The recording is the part to get right. Invoke happens immediately before the
// call and Return immediately after, so the recorded interval is as tight as the
// simulation can make it — a wider interval is not wrong, but it gives the
// checker more orderings to choose from and so weakens the test.
//
// An operation whose client never learned the outcome is deliberately left
// pending rather than recorded as a failure. A timed-out write may have reached
// every replica; claiming it did not would be a lie in the history, and linz
// would then report a violation of a promise nobody made.
func runClient(s *sim.Sim, h *linz.History, store KV, id int, opt Options) {
	r := s.Rand()
	for n := 0; n < opt.Ops; n++ {
		key := fmt.Sprintf("k%d", r.Intn(opt.Keys))
		if r.Bool(opt.WriteRatio) {
			// Values name their writer and its operation number, so a stale read
			// in a failure report says which write it should have seen.
			value := fmt.Sprintf("c%d-%d", id, n)
			inv := h.Invoke(id, linz.Put(key, value))
			if store.Put(key, value) {
				inv.Return(nil)
			}
		} else {
			inv := h.Invoke(id, linz.Get(key))
			if v, ok := store.Get(key); ok {
				inv.Return(v)
			}
		}
		s.Clock().Sleep(time.Duration(r.Duration(0, int64(opt.ThinkTime))))
	}
}

// injectFaults is the chaos actor: it partitions, heals, crashes and restarts on
// a schedule drawn from the seed.
//
// The timings are randomised rather than fixed because a fixed schedule explores
// one alignment between the faults and the workload, and the whole value of a
// seed sweep is that it explores many. The order is not randomised: heal before
// crash, so the cluster never has one replica isolated and another dead at the
// same time. That state leaves a single survivor, no quorum is reachable, and
// every operation in the window becomes an operation that never returned —
// which is a run that injected a great deal and observed almost nothing.
func injectFaults(c *Cluster, opt Options) {
	s := c.sim
	r := s.Rand()
	clock := s.Clock()

	if opt.Partition {
		clock.Sleep(millis(r, 20, 60))
		c.Isolate(r.Intn(opt.Cluster.Replicas))
		clock.Sleep(millis(r, 80, 200))
		c.Heal()
	}

	if opt.CrashRestart {
		clock.Sleep(millis(r, 20, 60))
		victim := r.Intn(opt.Cluster.Replicas)
		c.CrashReplica(victim)
		clock.Sleep(millis(r, 60, 150))
		c.RestartReplica(victim)
	}
}

// millis draws a duration in [lo, hi] milliseconds from the simulation's PRNG.
func millis(r *sim.Rand, lo, hi int64) time.Duration {
	return time.Duration(r.Duration(lo, hi)) * time.Millisecond
}

// Sweep runs one seed after another and hands each Outcome to visit, stopping
// early if visit returns false.
//
// Seeds are consecutive rather than random. A sweep that drew its own seeds
// would need a seed of its own, and "which seeds did CI actually try last night"
// is a question that should have an answer you can retype.
func Sweep(first sim.Seed, count int, opt Options, visit func(Outcome) bool) {
	for i := 0; i < count; i++ {
		if !visit(Run(first+sim.Seed(i), opt)) {
			return
		}
	}
}

// FirstViolation sweeps up to count seeds and returns the first one whose run
// the checker proved wrong, along with how many seeds that took.
//
// The count is what makes a failure to find one meaningful. "The bug did not
// reproduce" is useless without "in 64 seeds", because the reader's next
// question is always whether the search was serious.
//
// A simulation that fails to finish — a deadlock, a panic, an exhausted step
// budget — stops the sweep and is returned as-is. It is a bug in the harness
// rather than a finding about the store, and continuing past it would bury the
// evidence under seeds that could not have found anything either.
func FirstViolation(first sim.Seed, count int, opt Options) (out Outcome, tried int, found bool) {
	Sweep(first, count, opt, func(o Outcome) bool {
		tried++
		out = o
		if o.SimErr != nil {
			return false
		}
		found = o.Violated()
		return !found
	})
	return out, tried, found
}

// Summary renders the one-line-per-seed form the demo command prints.
func (o Outcome) Summary() string {
	switch {
	case o.SimErr != nil:
		return "SIMULATION FAILED"
	case o.Violated():
		return "VIOLATION"
	case o.Check.Status == linz.Unknown:
		return "UNKNOWN (search budget exhausted)"
	default:
		return fmt.Sprintf("ok   %3d ops (%d never returned), %d keys, %s virtual",
			o.Check.Operations, o.Check.Pending, o.Check.Partitions, o.VirtualTime.Round(time.Millisecond))
	}
}

// Story returns the simulation events that explain what happened to one key:
// every fault that was injected, and every moment a replica applied a write to
// that key, in the order they occurred.
//
// This is the view that turns a linearizability verdict into a diagnosis. The
// checker says a read returned a value no ordering allows; this says which
// replicas held what, and when — the two together are usually enough to name the
// bug without opening a debugger. It is how the retry bug documented on
// Client.Put was found.
//
// Trace.Tail is the general-purpose version and is the wrong tool here: a run
// continues for a long time after the violation, so its last events are the
// timers of operations that had nothing to do with it.
func (o Outcome) Story(key string) []string {
	if o.Trace == nil {
		return nil
	}
	var out []string
	for _, e := range o.Trace.Events() {
		if strings.Contains(e.Detail, "fault:") || strings.Contains(e.Detail, key+"=") {
			out = append(out, e.String())
		}
	}
	return out
}

// FailingKey returns the key whose sub-history could not be explained, or "" if
// the run produced no violation.
func (o Outcome) FailingKey() string {
	if o.Check.Failure == nil {
		return ""
	}
	key, _ := o.Check.Failure.PartitionKey.(string)
	return key
}

// KeysTouched returns the keys the run's history covers, sorted.
//
// Sorted because it is built by walking the history into a set, and a
// diagnostic that prints its keys in a different order on every run is one more
// thing a reader has to decide to ignore.
func (o Outcome) KeysTouched() []string {
	seen := make(map[string]bool)
	var out []string
	for _, op := range o.History.Operations() {
		in, ok := op.Input.(linz.KVInput)
		if !ok || seen[in.Key] {
			continue
		}
		seen[in.Key] = true
		out = append(out, in.Key)
	}
	sort.Strings(out)
	return out
}
