package kvstore

import (
	"fmt"
	"sort"
	"time"

	"github.com/avase33/lockstep/netsim"
	"github.com/avase33/lockstep/sim"
)

// KV is the contract both implementations claim to satisfy: a key-value store
// that behaves as if it were a single object handling one request at a time.
//
// Both methods report whether the client learned the outcome, which is a
// distinct thing from whether the operation succeeded. When a request times out,
// the write may have reached one replica, or all of them, or none — and nobody
// will ever know which. The harness records that case as an operation that never
// returned, which is exactly what linz's handling of pending operations is for.
// Returning an error instead would invite a caller to record a definite "it
// failed", and a history that claims a write did not happen when it did is a
// history that manufactures violations.
type KV interface {
	// Get returns the value at a key, with "" meaning absent, and ok=false if
	// the client never got an answer.
	Get(key string) (value string, ok bool)

	// Put stores a value, returning ok=false if the client never learned whether
	// it took effect.
	Put(key, value string) (ok bool)
}

// tag totally orders writes.
//
// The counter alone is not enough: two clients writing the same key
// concurrently both read the same highest counter and both add one, so the
// writer id is what keeps their tags distinct. Broken any other way — by arrival
// order, by replica — two different values could share a tag, and replicas would
// then disagree about which one won while every one of them believed it held the
// latest.
type tag struct {
	Counter uint64
	Writer  int
}

// after reports whether t supersedes o, comparing lexicographically.
func (t tag) after(o tag) bool {
	if t.Counter != o.Counter {
		return t.Counter > o.Counter
	}
	return t.Writer > o.Writer
}

func (t tag) String() string { return fmt.Sprintf("%d.%d", t.Counter, t.Writer) }

// versioned is a stored value together with the tag that wrote it. Its zero
// value means "this key has never been written", and it compares below every
// real tag, so a replica that has never heard of a key needs no special case
// anywhere.
type versioned struct {
	Value string
	Tag   tag
}

// The wire protocol: four message types, all plain values with no pointers in
// them. netsim hands the sender's body straight to the receiver, so a pointer
// body would be shared memory between two nodes that in production are two
// machines — the simulation would hide aliasing bugs instead of finding them.
//
// Every request carries a ReqID and every reply echoes it. That is not
// ceremony. This network duplicates messages and delivers them out of order, so
// a client waiting for a quorum genuinely does see replies to the request it
// made two round trips ago, and counting one of those toward the current quorum
// is the classic way to believe two replicas agreed when only one did.
type readReq struct {
	ReqID uint64
	Key   string
}

type readResp struct {
	ReqID uint64
	Val   versioned
}

type writeReq struct {
	ReqID uint64
	Key   string
	Val   versioned
}

type writeAck struct {
	ReqID uint64
}

// reply is what the client's collection loop needs from any response: the
// request it belongs to. Matching on a method rather than a type switch keeps
// the loop from having to know every message type in the protocol.
type reply interface{ requestID() uint64 }

func (r readResp) requestID() uint64 { return r.ReqID }
func (a writeAck) requestID() uint64 { return a.ReqID }

// ClusterConfig describes the cluster and the network it lives on. The zero
// value is a usable three-node cluster; see withDefaults for what each field
// becomes.
type ClusterConfig struct {
	// Replicas is how many copies of every key the cluster keeps. Three is the
	// smallest number for which a quorum is a real constraint rather than
	// "everyone", and the smallest for which one node can be lost without
	// stopping writes.
	Replicas int

	// Timeout bounds one round of requests. It must be comfortably larger than
	// two times the network's MaxLatency, or every round times out and the
	// workload degenerates into a history of operations that never returned —
	// which linz will happily declare linearizable, because nothing was ever
	// observed. A test that passes for that reason is worse than no test.
	Timeout time.Duration

	// Retries is how many times a client re-attempts a round that timed out
	// before giving up and reporting ok=false. Zero means the default of 2;
	// negative means none.
	//
	// Retrying matters for the quality of the history, not for the store's
	// correctness: an operation that gives up becomes an operation that never
	// returned, and those constrain the checker far less than ones that did. A
	// couple of retries keeps most operations observable through a partition and
	// a crash, which is where the interesting orderings are.
	Retries int

	// ApplyDelay bounds the time a replica spends making a write durable before
	// it acknowledges — the fsync it must not skip. Drawn per write from
	// Sim.Rand. Zero means the default; negative means none.
	//
	// It is not decoration. It puts a window between "the request arrived" and
	// "the acknowledgement was sent" that a crash can land inside, and it lets
	// replicas fall behind each other by different amounts, which is the state
	// the single-replica read bug needs in order to be visible.
	ApplyDelay time.Duration
}

func (c ClusterConfig) withDefaults() ClusterConfig {
	if c.Replicas <= 0 {
		c.Replicas = 3
	}
	if c.Timeout <= 0 {
		c.Timeout = 120 * time.Millisecond
	}
	switch {
	case c.Retries == 0:
		c.Retries = 2
	case c.Retries < 0:
		c.Retries = 0
	}
	switch {
	case c.ApplyDelay == 0:
		c.ApplyDelay = 3 * time.Millisecond
	case c.ApplyDelay < 0:
		c.ApplyDelay = 0
	}
	return c
}

// replica is one server's durable state.
//
// It is separate from the actor that serves it because the actor dies on a crash
// and this does not: writes are on disk before they are acknowledged, so a
// restarted replica comes back holding everything a quorum ever accepted from
// it. See the package documentation for why the whole quorum argument collapses
// if that is not true.
type replica struct {
	id   int
	data map[string]versioned
}

// Cluster is the replicated store: the replicas, the network they sit on, and
// the faults a test can inject into them.
//
// Like sim.Sim and netsim.Network it needs no internal locking and must not be
// touched from an unmanaged goroutine, for the same reason: every method here
// runs while its caller holds the scheduler's baton, so there is never a second
// actor inside it.
type Cluster struct {
	sim *sim.Sim
	net *netsim.Network
	cfg ClusterConfig

	reps []*replica
	// addrs maps replica index to the node name its current incarnation answers
	// on, and gen counts how many times each has been restarted. The indirection
	// exists because a restart cannot reuse the old name; see the package
	// documentation.
	addrs []string
	gen   []int

	// faults is a human-readable log of what was injected, in the order it
	// happened. It is what lets a test assert that the run really did partition
	// and crash something rather than quietly doing nothing, which is the way
	// fault-injection tests usually rot.
	faults []string

	// diverged latches once two replicas have been seen holding different values
	// for the same key. See Diverged.
	diverged bool
}

// NewCluster starts a replicated store on an existing simulation and network.
//
// Each replica joins the network and gets an actor of the same name, which is
// what lets netsim discover that a crashed replica is dead and stop delivering
// to it. Breaking that correspondence would make the network deliver messages
// into a receiver that will never run again, and they would not be dropped —
// they would be swallowed.
func NewCluster(s *sim.Sim, net *netsim.Network, cfg ClusterConfig) *Cluster {
	cfg = cfg.withDefaults()
	c := &Cluster{sim: s, net: net, cfg: cfg}
	for i := 0; i < cfg.Replicas; i++ {
		rep := &replica{id: i, data: make(map[string]versioned)}
		c.reps = append(c.reps, rep)
		c.gen = append(c.gen, 1)
		c.addrs = append(c.addrs, replicaAddr(i, 1))
		c.start(rep)
	}
	return c
}

// replicaAddr names one incarnation of a replica. The generation is in the name
// so that a trace or a deadlock report says which incarnation it means.
func replicaAddr(id, gen int) string {
	if gen == 1 {
		return fmt.Sprintf("replica-%d", id)
	}
	return fmt.Sprintf("replica-%d#g%d", id, gen)
}

// start joins the current incarnation of a replica to the network and runs its
// serve loop as an actor of the same name.
func (c *Cluster) start(rep *replica) {
	addr := c.addrs[rep.id]
	node := c.net.Join(addr)
	c.sim.Go(addr, func() { c.serve(rep, node) })
}

// serve is a replica's whole life: answer reads from its own copy, apply writes
// that supersede what it holds, acknowledge everything.
//
// The loop blocks in Recv, which is how a real server waits and is safe here in
// a way that blocking on a Go channel is not — the scheduler knows this actor is
// waiting and will name it if the run deadlocks. It ends when the network
// closes and the inbox drains.
func (c *Cluster) serve(rep *replica, node *netsim.Node) {
	for {
		m, ok := node.Recv()
		if !ok {
			return
		}
		switch req := m.Body.(type) {
		case readReq:
			// Reads are served from whatever this replica happens to hold. That
			// is honest, and it is the fact the buggy read path forgets: a
			// replica is not required to be up to date, only to be part of
			// enough quorums.
			node.Send(m.From, readResp{ReqID: req.ReqID, Val: rep.data[req.Key]})

		case writeReq:
			// Durability before acknowledgement. Sleeping here rather than after
			// the apply is deliberate: a crash during this window loses the write
			// entirely, and that must be safe, because the client has not been
			// told anything yet.
			c.sim.Clock().Sleep(c.drawApplyDelay())
			if req.Val.Tag.after(rep.data[req.Key].Tag) {
				rep.data[req.Key] = req.Val
				c.sim.Record("replica %d applied %s=%q tag=%v", rep.id, req.Key, req.Val.Value, req.Val.Tag)
				c.noteDivergence()
			}
			// Acknowledged even when the write was superseded by a newer tag.
			// The client asked for its value to be at least as recent as this
			// one, and a replica holding something newer satisfies that; refusing
			// would stall a write that has already been overtaken, forever.
			node.Send(m.From, writeAck{ReqID: req.ReqID})

		default:
			panic(fmt.Sprintf("kvstore: replica %d got unknown message %T", rep.id, m.Body))
		}
	}
}

func (c *Cluster) drawApplyDelay() time.Duration {
	if c.cfg.ApplyDelay <= 0 {
		return 0
	}
	return time.Duration(c.sim.Rand().Duration(0, int64(c.cfg.ApplyDelay)))
}

// quorum is the smallest majority of replicas.
//
// Reads and writes both use it, and that is the entire safety argument: two
// majorities of the same set always share a member, so any quorum read
// intersects the quorum that accepted the last acknowledged write. Lower either
// one and the store stops being able to tell you anything.
func (c *Cluster) quorum() int { return c.cfg.Replicas/2 + 1 }

// Addrs returns the node name of each replica's current incarnation, indexed by
// replica id.
//
// A fresh slice, not the cluster's own: a client holds this across a round of
// requests, and a restart during that round must not silently redirect messages
// the client already believes it addressed.
func (c *Cluster) Addrs() []string {
	out := make([]string, len(c.addrs))
	copy(out, c.addrs)
	return out
}

// Faults returns the fault-injection log, in the order things were injected.
//
// Exported so a test can assert the run actually did something. A fault
// injector that silently stops firing — a misspelt field, a partition healed one
// line too early — turns a chaos test into a green test that exercises the happy
// path, and nothing about the output looks different.
func (c *Cluster) Faults() []string {
	out := make([]string, len(c.faults))
	copy(out, c.faults)
	return out
}

// Diverged reports whether two replicas ever held different values for the same
// key at any point in the run.
//
// This is the state that makes single-replica reads unsafe, and it is worth
// asserting on its own: if a change to the workload or the fault rates stopped
// producing divergence, the store would look correct for a reason that has
// nothing to do with its read path, and the headline test would go green having
// quietly stopped testing anything.
//
// "Ever", not "at the end". A quorum read writes its answer back, so a run of
// the fixed store leaves the replicas in agreement however violently they
// disagreed on the way — an end-of-run comparison would report no divergence
// precisely when the store was working hardest.
func (c *Cluster) Diverged() bool { return c.diverged }

// noteDivergence records whether the replicas currently disagree. Called after
// every applied write, and cheap once the answer is yes, which it usually
// becomes early.
//
// The key union is sorted before it is walked — not because the answer depends
// on the order, but because "it happens not to matter here" is not a rule that
// survives the next edit.
func (c *Cluster) noteDivergence() {
	if c.diverged {
		return
	}
	seen := make(map[string]bool)
	var keys []string
	for _, rep := range c.reps {
		for k := range rep.data {
			if !seen[k] {
				seen[k] = true
				keys = append(keys, k)
			}
		}
	}
	sort.Strings(keys)

	for _, k := range keys {
		first := c.reps[0].data[k].Value
		for _, rep := range c.reps[1:] {
			if rep.data[k].Value != first {
				c.diverged = true
				return
			}
		}
	}
}

func (c *Cluster) logFault(format string, args ...any) {
	s := fmt.Sprintf(format, args...)
	c.faults = append(c.faults, s)
	c.sim.Record("fault: %s", s)
}

// Isolate cuts one replica off from every other node on the network, in both
// directions, until Heal.
//
// This is the single most productive fault for a replicated store, because of
// what it leaves behind. Writes carry on succeeding — a quorum of the survivors
// is still a quorum — so the isolated replica quietly falls arbitrarily far
// behind, and when the network heals it is back in service, answering reads
// promptly and confidently with data that is minutes old. A store with no read
// repair never notices. That is a production incident, not a thought experiment.
//
// The peer list comes from Network.Nodes, which sorts, so which links are cut
// does not depend on Go's map iteration order.
func (c *Cluster) Isolate(id int) {
	victim := c.addrs[id]
	var others []string
	for _, n := range c.net.Nodes() {
		if n != victim {
			others = append(others, n)
		}
	}
	c.net.Partition([]string{victim}, others)
	c.logFault("isolated %s from the rest of the cluster", victim)
}

// Heal restores full connectivity. Messages dropped during the partition stay
// dropped; catching up is the store's problem, which is the point.
func (c *Cluster) Heal() {
	c.net.Heal()
	c.logFault("healed the network")
}

// CrashReplica kills a replica's serving process where it stands.
//
// The actor stops mid-operation holding whatever it held, and netsim stops
// delivering to it: messages already on the wire toward it are dropped with
// reason "crashed" rather than handed to a process that will never read them.
// Its data survives, because writes were durable before they were acknowledged.
//
// The crash is permanent for that incarnation. Bring the replica back with
// RestartReplica.
func (c *Cluster) CrashReplica(id int) {
	addr := c.addrs[id]
	c.sim.Crash(addr)
	c.logFault("crashed %s", addr)
}

// RestartReplica brings a crashed replica back as a new incarnation, reloading
// the data it had made durable.
//
// The new incarnation answers on a new address, because sim.Crash is permanent
// and both Sim.Go and Network.Join reject a reused name — see the package
// documentation for why that is the right model rather than a workaround.
// Clients find it through Addrs, which is the indirection every system with
// ephemeral addressing already has.
//
// The new incarnation starts with an empty inbox, which is what makes this a
// restart rather than a resumption: requests sent to the dead process are gone,
// exactly as they would be in a system where the socket died with it.
func (c *Cluster) RestartReplica(id int) {
	c.gen[id]++
	c.addrs[id] = replicaAddr(id, c.gen[id])
	c.start(c.reps[id])
	c.logFault("restarted replica %d as %s", id, c.addrs[id])
}

// Client is the half of the store that both implementations share: the
// transport, the request numbering, and the write path.
//
// One per client actor. Sharing a Client between actors would interleave two
// conversations on one inbox, and each would consume the other's replies —
// a fault that looks exactly like a flaky network and is not one.
type Client struct {
	cl   *Cluster
	node *netsim.Node
	id   int
	next uint64
}

// Client joins a client endpoint to the network and returns a handle for
// talking to the cluster.
//
// The id becomes the client's tag writer id, so it must be unique across
// clients: two clients sharing one would be able to produce identical tags for
// different values, and the replicas would silently disagree about which of them
// won.
func (c *Cluster) Client(id int) *Client {
	return &Client{cl: c, node: c.net.Join(fmt.Sprintf("client-%d", id)), id: id}
}

// Name returns the client's node name, which is also its actor name.
func (c *Client) Name() string { return c.node.Name() }

func (c *Client) nextReq() uint64 {
	c.next++
	return c.next
}

// round sends one request to every replica and collects replies from `want`
// distinct replicas, or gives up when the round's deadline passes.
//
// Three details in here are the difference between a client that works on this
// network and one that appears to:
//
// Replies are matched on the request id, because a duplicated or badly delayed
// reply to an earlier request is a normal event here, not a corruption.
//
// Replies are counted per replica, because the network duplicates messages, and
// two copies of one replica's answer is not two replicas agreeing. Getting this
// wrong turns a quorum of two into a quorum of one, which is invisible until it
// costs you a write.
//
// The deadline is created once, before the loop. Recreating it per iteration
// would restart the timeout every time an irrelevant message arrived, so a
// client that was being kept busy by stale duplicates would wait forever.
func (c *Client) round(reqID uint64, body any, want int) ([]reply, bool) {
	s := c.cl.sim
	for _, addr := range c.cl.Addrs() {
		c.node.Send(addr, body)
	}

	deadline := s.Clock().After(c.cl.cfg.Timeout)
	seen := make(map[string]bool)
	got := make([]reply, 0, want)
	for len(got) < want {
		i, v, ok := sim.Select(s, c.node.Inbox(), deadline)
		if i != 0 {
			return got, false // the deadline fired first
		}
		if !ok {
			return got, false // the network shut down under us
		}
		m := v.(netsim.Message)
		r, isReply := m.Body.(reply)
		if !isReply || r.requestID() != reqID || seen[m.From] {
			continue
		}
		seen[m.From] = true
		got = append(got, r)
	}
	return got, true
}

// attempt runs one round, retrying it until it succeeds or the retry budget is
// spent.
//
// Each attempt gets a fresh request id, never a reused one. Replies to the
// abandoned attempt are still in flight — that is why it timed out — and
// counting one of those toward the retry's quorum would let a single replica
// answer for two.
//
// What must NOT be rebuilt on a retry is the payload, which is why this takes
// the body from a closure the caller controls rather than rebuilding it here.
// See Put.
func (c *Client) attempt(want int, build func(reqID uint64) any) ([]reply, bool) {
	for i := 0; i <= c.cl.cfg.Retries; i++ {
		id := c.nextReq()
		if replies, ok := c.round(id, build(id), want); ok {
			return replies, true
		}
	}
	return nil, false
}

// Put writes a value and returns once a quorum of replicas has it durably.
//
// Both implementations use this, unchanged. That is deliberate: the two stores
// differ in exactly one method, so a violation found against one and not the
// other is a fact about the read path and cannot be explained away as a
// difference in the workload.
//
// It is a two-phase write, and neither phase is optional.
//
// The first phase picks a tag that beats every tag already out there. Without
// it, a write could land behind one that finished before it even started and be
// silently discarded as stale. Asking a quorum for the highest tag it has seen
// is what guarantees the new one is bigger: any completed write is on a quorum,
// and any two quorums intersect.
//
// The second phase is retried with THE SAME TAG, and that is the subtle part.
// Deriving a fresh tag on retry looks harmless — it is the same value, going to
// the same key — and it is not. The first attempt may already have reached some
// replicas, so the write ends up present at two different points in the tag
// order, and another client's write can complete in between. One client-visible
// Put then takes effect twice, and no ordering of the history explains it. This
// example had that bug, linz found it on seed 0x00000008, and it took a trace of
// the applied tags to see it. A retried request has to carry the payload it
// carried the first time; re-deriving anything is how idempotence is lost.
func (c *Client) Put(key, value string) bool {
	q := c.cl.quorum()

	replies, ok := c.attempt(q, func(id uint64) any { return readReq{ReqID: id, Key: key} })
	if !ok {
		return false
	}
	val := versioned{Value: value, Tag: tag{Counter: highest(replies).Tag.Counter + 1, Writer: c.id}}

	_, ok = c.attempt(q, func(id uint64) any { return writeReq{ReqID: id, Key: key, Val: val} })
	return ok
}

// highest returns the newest versioned value among a round's replies.
//
// The replies are walked in arrival order, which is a slice and therefore
// deterministic. Ties are impossible between different values — that is what the
// writer id in a tag is for — so first-wins on an exact tie is safe: the two
// replies are the same write.
func highest(replies []reply) versioned {
	var best versioned
	for _, r := range replies {
		if resp, ok := r.(readResp); ok && resp.Val.Tag.after(best.Tag) {
			best = resp.Val
		}
	}
	return best
}

// Store is the buggy implementation: quorum writes, single-replica reads.
//
// It is the store this example exists to convict. Everything about it is
// correct except Get, and Get is wrong in the way that reads always go wrong —
// by being made cheap.
type Store struct{ *Client }

// QuorumStore is the correct implementation: the same store with an ABD read.
type QuorumStore struct{ *Client }

var (
	_ KV = (*Store)(nil)
	_ KV = (*QuorumStore)(nil)
)

// Get asks every replica and returns the first answer that arrives. This is the
// bug.
//
// It looks like a latency optimisation and it is one — it saves a round trip to
// the second-slowest replica, on the path that carries most of the traffic. What
// it costs is the only property the store was worth having: a quorum write is
// acknowledged when a MAJORITY has applied it, so the replica that answers
// fastest is not necessarily one of them. When it is not, this returns a value
// that was overwritten before the read was even invoked, and no ordering of the
// two operations explains that.
//
// There is no read repair either, so a replica that fell behind during a
// partition stays behind indefinitely and keeps answering reads with data from
// before the split.
func (s *Store) Get(key string) (string, bool) {
	replies, ok := s.attempt(1, func(id uint64) any { return readReq{ReqID: id, Key: key} })
	if !ok {
		return "", false
	}
	return replies[0].(readResp).Val.Value, true
}

// Get reads from a quorum, writes the winning value back to a quorum, and only
// then returns it. This is ABD, and it is correct.
//
// The first phase is the obvious half: a read quorum intersects the write quorum
// of every acknowledged write, so the highest tag it sees is at least as new as
// the last write that completed.
//
// The second phase is the half that gets left out, and leaving it out is a real
// bug rather than a missing optimisation. Consider a write that reached one
// replica and then died with its client. It is not "in" the store — no quorum
// holds it — but it is not gone either. A quorum read that happens to include
// that replica returns the new value; the next quorum read, asking two others,
// returns the old one. Two reads with no write between them, and the second went
// backwards. Writing the value back before returning it forbids that: nothing is
// ever reported to a client until a quorum holds it, so every later read is
// guaranteed to find it.
//
// The cost is that reads write, which is why people skip it.
func (q *QuorumStore) Get(key string) (string, bool) {
	n := q.cl.quorum()

	replies, ok := q.attempt(n, func(id uint64) any { return readReq{ReqID: id, Key: key} })
	if !ok {
		return "", false
	}
	best := highest(replies)

	// Written back under its original tag, for the reason spelled out in Put:
	// a retry that changed the tag would give one read two positions in the
	// order.
	if _, ok := q.attempt(n, func(id uint64) any { return writeReq{ReqID: id, Key: key, Val: best} }); !ok {
		return "", false
	}
	return best.Value, true
}
