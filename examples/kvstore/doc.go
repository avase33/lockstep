// Package kvstore is lockstep's headline demonstration: a replicated key-value
// store containing a bug that people really ship, and a test that finds it,
// names the seed that found it, and reproduces it byte for byte on demand.
//
// # The bug
//
// The store replicates every key across three nodes. A write is sent to all of
// them and acknowledged as soon as a quorum — two of the three — has applied it.
// That part is right, and it is the part everybody gets right, because it is the
// part the design document talks about.
//
// The read path is where it goes wrong:
//
//	Store.Get sends the read to every replica and returns the FIRST reply.
//
// This is not a strawman. It is the natural thing to write once writes are
// working: reads outnumber writes, a quorum read costs an extra round trip to
// the slowest of two machines, and "ask everyone, take whoever answers first"
// is both obvious and measurably faster. Systems ship with exactly this. The
// usual phrasing in the postmortem is "we added read replicas to scale reads".
//
// It is also wrong, and wrong in a way that no amount of staring finds. A write
// acknowledged by replicas 0 and 1 has not necessarily reached replica 2 — that
// is the entire point of a quorum. If replica 2 answers a later read first, the
// client is told a value that was already overwritten before its read was even
// invoked. The write returned, then the read started, and the read returned
// stale data. There is no ordering of those two operations that a single
// correct object could have produced, so the history is not linearizable, and
// linz proves it rather than guessing.
//
// The failure needs a replica to be behind at the moment a read reaches it,
// which in production means a slow disk, a garbage-collection pause, a network
// hiccup, or a node that has just come back from a partition. In a normal test
// suite it needs all the stars to align and so it never happens. Under lockstep
// it happens on the first seed, because the simulator's job is to arrange
// exactly those stars, and its other job is to hand you the seed that did.
//
// # The fix
//
// QuorumStore.Get is the same store with a correct read path, which is the ABD
// algorithm (Attiya, Bar-Noy and Dolev, 1995):
//
//  1. Read from a quorum and take the value carrying the highest tag.
//  2. Write that value back to a quorum.
//  3. Only then return it.
//
// Step 1 alone is not enough, and the reason is worth understanding, because
// "quorum read" is widely believed to be sufficient on its own. Suppose a write
// dies halfway: it reaches replica 0 and its client gives up. A quorum read that
// happens to ask replicas 0 and 1 sees the new value and returns it. A later
// quorum read that asks replicas 1 and 2 does not, and returns the old one. Two
// reads, no write in between, and the second one goes backwards in time. Step 2
// is what forbids that: a value is not returned to anyone until it is on a
// quorum, so every subsequent read is guaranteed to intersect it.
//
// Both implementations share the write path and the transport. They differ in
// exactly one method, which is the whole argument of this example: the harness
// rejects Store.Get and accepts QuorumStore.Get, so it is detecting the defect
// rather than complaining about the workload.
//
// # What this package demonstrates
//
// Run finishes a full simulation — three replicas, four clients, latency, drops,
// duplicates, a network partition and a node crash with restart — in a few
// milliseconds of real time, and returns a linz verdict on what happened.
// Everything about that run is a function of the seed, so:
//
//   - TestSingleReplicaReadsAreNotLinearizable sweeps seeds until one produces a
//     violation and prints the seed, the report and the repro command.
//   - TestQuorumReadsAreLinearizable runs the fixed store over the same
//     workload across many seeds and requires every one to be clean. This is the
//     test that makes the first one mean anything: a checker that rejects
//     everything has found nothing.
//   - TestReproducibleFromSeed replays the violating seed ten times and requires
//     an identical trace hash and an identical failure report every time.
//
// # The determinism rules, as they apply here
//
// This package obeys the rules in package sim without exception, because an
// example that broke them would discredit the thing it exists to sell:
//
//   - Every random choice — which key, which operation, how long to think, which
//     replica to isolate, when to crash it — comes from Sim.Rand.
//   - Every wait goes through Clock.Sleep, Clock.After or a sim Chan. There is
//     no time.Sleep, no time.Now and no native Go channel anywhere in it.
//   - Maps are only ever read by key. The one place a set of nodes must be
//     enumerated (Cluster.Isolate) takes it from Network.Nodes, which sorts.
//   - The workload's history is recorded with linz.History, whose counter
//     increments under the same lock that appends the operation, so the recorded
//     real-time order is exactly the order the scheduler produced.
//
// # Modelling a restart
//
// sim.Crash is permanent by design: there is no way to un-crash an actor, and
// netsim learns that a node is dead by asking Sim.Crashed about the actor whose
// name matches the node's. A restart therefore cannot reuse the old name — both
// Sim.Go and Network.Join reject duplicates — so Cluster.RestartReplica brings
// the replica back as a new incarnation at a new address, and Cluster.Addrs maps
// replica index to its current one.
//
// That is not a workaround dressed up as a design. It is what a restart looks
// like in any system with ephemeral addressing, and it gives the right
// semantics for free: messages still in flight to the dead incarnation are
// dropped with reason "crashed" instead of being delivered to a corpse, and the
// new incarnation starts with an empty inbox rather than inheriting a backlog of
// requests that were sent to a process that no longer existed.
//
// What survives the crash is the replica's data, because writes are durable
// before they are acknowledged. That is not a convenience: it is the assumption
// the whole quorum argument rests on. A replica that came back empty could
// silently destroy a write that a quorum had already accepted, and then even
// QuorumStore would be non-linearizable — the bug would be in the storage
// engine rather than the read path, and this example would be demonstrating the
// wrong thing.
package kvstore
