# lockstep

Deterministic simulation testing for Go. Find a concurrency bug once, reproduce
it forever from a seed.

```
FAIL: linearizability violated in 5 of 8 seeds
  seed:   0x00000001
  trace:  f3d9a89ec7d4cfa8
  faults: isolated replica-1 from the rest of the cluster; healed the network;
          crashed replica-0; restarted replica 0 as replica-0#g2
  repro:  go run ./examples/kvstore/cmd -seed=0x00000001
```

That last line is the point. A bug that used to be "fails sometimes in CI"
becomes a command.

---

## The problem

Distributed bugs are the ones that ruin your week. A race that fires once in ten
thousand runs. A partition arriving at exactly the wrong moment. A retry that
duplicates a write only when the ack is lost *after* the write landed.

You can't reproduce them, so you can't fix them, so you add a retry and hope.

FoundationDB solved this by running their entire database inside a simulator
with deterministic time, scheduling and I/O. TigerBeetle does the same.
Antithesis sells it. If you're an ordinary Go team, there hasn't been a good
open-source way to get it.

## How it works

Your code runs against small interfaces — a clock, a network. In production
those are real. Under test lockstep supplies fakes driven by **one seeded
PRNG**, runs every goroutine on a **deterministic scheduler**, and injects
latency, drops, duplicates, partitions and crashes.

Same seed, same code, same run. Every time, on every machine.

```go
s := sim.New(sim.Config{Seed: 0x4f2a91c3})
net := netsim.New(s, netsim.Config{
    MinLatency: time.Millisecond, MaxLatency: 20 * time.Millisecond,
    DropRate: 0.04, DuplicateRate: 0.06,
})

for i := 0; i < 3; i++ {
    node := net.Join(fmt.Sprintf("replica-%d", i))
    s.Go(node.Name(), func() { replica(s, node) })
}
s.Go("chaos", func() {
    s.Clock().Sleep(46 * time.Millisecond)
    net.Partition([]string{"replica-1"}, []string{"replica-0", "replica-2"})
    s.Clock().Sleep(128 * time.Millisecond)
    net.Heal()
})

res := s.Run()
if res.Failed() {
    t.Fatal(res.Error())   // prints the seed and the repro command
}
```

Virtual time is free: a 24-hour simulated timeout costs microseconds, so you can
finally test the retry path nobody tests.

## The demo

`examples/kvstore` is a replicated key-value store with a bug a competent
engineer would actually write: **writes go to a quorum, reads come from whichever
replica answers first.** No quorum read, no read repair.

That isn't linearizable — a read can hit a replica that missed the latest write
and return a stale value, even though the write was acknowledged before the read
was invoked.

```
$ go run ./examples/kvstore/cmd
lockstep: sweeping 8 seeds against kvstore (single-replica reads)
  cluster: 3 replicas, quorum of 2, 120ms request timeout, 2 retries
  clients: 4, each doing 12 operations over 3 keys, 34% writes
  network: 1ms-20ms latency, 4% loss, 6% duplication, one replica isolated
           then healed, one replica crashed then restarted

  seed 0x00000001  VIOLATION
  seed 0x00000002  VIOLATION
  seed 0x00000003  VIOLATION
  seed 0x00000004  VIOLATION
  seed 0x00000005  VIOLATION
  seed 0x00000006  ok    48 ops (0 never returned), 3 keys, 618ms virtual
  seed 0x00000007  ok    48 ops (0 never returned), 3 keys, 757ms virtual
  seed 0x00000008  ok    48 ops (0 never returned), 3 keys, 786ms virtual

8 of 8 seeds checked in 68ms
```

The checker doesn't just say "broken". It shows the longest ordering that *did*
work and the exact operation that couldn't be placed:

```
  not linearizable: 48 operations in 3 partitions (largest 21, max concurrency 3)

  no ordering of these operations can explain what the clients observed
  on key "k0" (21 operations touch it).

    the longest ordering that works covers 8 of 21 operations:
       ...
       5. put("k0", "c3-1") -> ok     [client 3, invoked 9, returned 20]
       6. get("k0") -> c3-1           [client 1, invoked 24, returned 28]
       7. get("k0") -> c3-1           [client 1, invoked 33, returned 35]
       8. get("k0") -> c3-1           [client 3, invoked 34, returned 37]

    state after those operations: {k0: "c3-1"}

    one of these had to happen next, and none of them could:
        get("k0") -> (absent)         [client 1, invoked 36, returned 42]
            impossible in the state above
```

And `-story` replays what the cluster actually did:

```
   159         46ms  chaos      fault: isolated replica-1 from the rest of the cluster
   192  60.143016ms  replica-2  replica 2 applied k0="c3-1" tag=1.3
   198  67.386845ms  replica-0  replica 0 applied k0="c3-1" tag=1.3
   391        174ms  chaos      fault: healed the network
   436        195ms  chaos      fault: crashed replica-0
   546 243.25583ms   replica-2  replica 2 applied k0="c0-4" tag=2.0
   672        285ms  chaos      fault: restarted replica 0 as replica-0#g2
```

replica-1 was isolated when `c3-1` was written, so it never got it. After the
heal it answered a read first, with nothing.

The same file has `QuorumStore`, which does full ABD — quorum read, take the
highest tag, write it back — and survives **3,000 consecutive seeds with zero
violations**. That half matters as much as the first: a checker that only ever
fails is a checker nobody can trust.

## Packages

| Package | What it is |
|---|---|
| `sim` | The deterministic core: scheduler, virtual clock, seeded PRNG, channels, select |
| `netsim` | Simulated network: latency, drops, duplicates, symmetric and one-way partitions |
| `linz` | Linearizability checker (Wing & Gong with P-compositionality), register and KV models |
| `examples/kvstore` | The demo above |

## Determinism is tested, not asserted

The guarantee is worthless if nobody checks it, so lockstep checks it:

- The same seed produces a **byte-identical trace hash** across 20 runs.
- It holds across `GOMAXPROCS` of 1, 2, 4 and 8 — so a laptop and a CI box agree.
- 40 seeds produce at least 20 distinct schedules, proving it actually explores
  rather than replaying one path.
- `netsim`'s determinism tests were mutation-tested: seeding the drop decision
  from `math/rand` instead of the seeded source is caught immediately.

## How it stays deterministic

At most **one** actor goroutine is runnable at any instant. The scheduler holds a
baton; an actor runs until it sleeps, sends, receives or yields, then hands it
back. The Go runtime is never asked to choose between two runnable actors, so its
nondeterminism is removed from the equation rather than fought.

The cost, stated plainly: there is no parallelism, so lockstep cannot find bugs
that need two instructions to genuinely execute at the same instant on two cores.
It finds *ordering* bugs. Use `-race` for the other kind. This is the same trade
FoundationDB and TigerBeetle make.

**What breaks determinism** — each of these has cost someone a week:

- `time.Now`, `time.Sleep`, `time.After` → use `sim.Clock`
- `math/rand`'s global functions, `crypto/rand` → use `sim.Rand`
- Iterating a map and acting on the order → sort the keys first
- Goroutines not started with `sim.Go`
- Real network or disk I/O
- `sync.WaitGroup` or native channels between actors → use `sim.Chan`

## Limits, honestly

- **No parallelism.** See above.
- **Linearizability checking is exponential in concurrency per key.** Measured:
  100 operations on one register take 43 µs at 2 concurrent clients, 1.3 ms at 8,
  43 ms at 12, 190 ms at 14, 7.4 s at 18. Beyond ~18 concurrent clients per key
  the budget is exhausted and the verdict is `Unknown` — never a false pass.
  Length is cheap: 100,000 sequential operations check in 49 ms.
- **Each in-flight message is a registered actor**, and the scheduler rescans
  every actor per step, so very large message counts go quadratic.
- **A crashed actor's goroutine parks forever** rather than being killed, because
  Go provides no way to kill a goroutine. Bounded and cheap, but it is a leak.
- `sim.Crash` is permanent and names are unique, so a restarted node needs a new
  name. `examples/kvstore` models this as a new incarnation at a new address.

## Bugs it found while being built

Not a marketing section — these are the reason to believe the rest.

- **In lockstep's own scheduler.** When a crashed actor's timer was the earliest
  pending one, the clock advanced but reported no progress, so `Run` declared a
  spurious deadlock with live timers still queued. Crashing a node asleep on a
  heartbeat is the most common thing a crash test does, so it fired constantly
  and blamed the system under test. Found while writing `netsim`; fixed with
  regression coverage.
- **In an earlier `Config.Trace bool`** documented as defaulting to true, which
  Go zero-valued to false. Tracing was silently off, and the determinism test
  compared two empty hashes and passed. Now `DisableTrace`, so the safe state is
  the zero value.
- **Twice in the demo's own "correct" store.** The first version derived a new
  higher tag on retry, so one client-visible `Put` took effect at two points in
  the tag order. The second used quorum reads without write-back, which is still
  not linearizable if a write dies after reaching one replica. `linz` caught
  both.

## Install

```
go get github.com/avase33/lockstep
```

Go 1.24 or later. No dependencies outside the standard library.

## Prior art

FoundationDB's simulation, TigerBeetle's VOPR, Antithesis, and Jepsen's Knossos
and Porcupine for linearizability checking. lockstep packages the technique as a
library you can point at ordinary Go code.

## Licence

MIT.
