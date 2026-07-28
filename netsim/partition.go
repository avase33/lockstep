package netsim

import (
	"fmt"
	"sort"
	"strings"
)

// Partition cuts every link between two groups of nodes, in both directions,
// until Heal.
//
// Partitions are additive. Calling Partition twice with different groups leaves
// both cuts in place, which is how you build a three-way split or an
// asymmetric topology out of a primitive that only knows about two sets. Heal
// removes everything at once; there is deliberately no "un-partition just this
// pair", because a test that needs that level of surgery is describing a
// topology, and describing it as a sequence of cuts from a known-healed state is
// far easier to read six months later.
//
// Nodes inside a group can still talk to each other, and nodes not named in
// either group are unaffected. Unknown names panic: a partition against a
// misspelled node silently does nothing, the cluster keeps converging, and the
// test passes while asserting nothing at all. That failure is expensive enough
// to be worth a panic.
func (nw *Network) Partition(groupA, groupB []string) {
	nw.cut(groupA, groupB)
	nw.cut(groupB, groupA)
	nw.sim.Record("net partition %s | %s", renderGroup(groupA), renderGroup(groupB))
}

// PartitionOneWay blocks traffic from src to dst while leaving dst to src
// working.
//
// This is the fault worth having and the reason a symmetric-only API is not
// enough. A symmetric partition is easy mode: both sides notice they are alone
// and behave. A one-way partition is where consensus implementations go wrong,
// because the two sides disagree about whether anything is broken. A leader
// whose heartbeats still land but whose acknowledgements never come back holds
// its term forever while the followers time out and elect around it — two
// leaders, no error anywhere, and it reproduces on demand here.
//
// Most test harnesses cannot express this at all, because they model a partition
// as a set of groups rather than as a set of directed links. This one stores
// links.
func (nw *Network) PartitionOneWay(src, dst []string) {
	nw.cut(src, dst)
	nw.sim.Record("net partition-oneway %s -> %s", renderGroup(src), renderGroup(dst))
}

// cut blocks every directed link from one group to another.
func (nw *Network) cut(from, to []string) {
	for _, a := range from {
		nw.mustNode(a, "Partition")
		for _, b := range to {
			nw.mustNode(b, "Partition")
			if a == b {
				// A node is never partitioned from itself. Blocking loopback
				// would be an artefact of listing a node in both groups, not a
				// fault anyone means to inject.
				continue
			}
			nw.blocked[link{a, b}] = true
		}
	}
}

// Heal removes every partition, symmetric and one-way alike, restoring full
// connectivity.
//
// Total rather than selective, because the interesting question after a
// partition is almost always "does the cluster reconverge once the network is
// fine again", and a partial heal makes it ambiguous which cut was responsible
// for whatever happens next.
//
// Messages dropped during the partition stay dropped. There is no replay queue,
// because a network that redelivers everything it withheld is not a network any
// distributed system is designed against — recovering from that loss is the
// system under test's job, and quietly doing it for them would make the test
// pass for the wrong reason.
func (nw *Network) Heal() {
	if len(nw.blocked) == 0 {
		return
	}
	nw.blocked = make(map[link]bool)
	nw.sim.Record("net heal")
}

// Blocked reports whether the network is currently refusing traffic from one
// node to another. Directed: Blocked(a, b) and Blocked(b, a) can disagree, and
// checking both is how a test proves a partition really is one-way.
func (nw *Network) Blocked(from, to string) bool {
	return nw.blocked[link{from, to}]
}

// Partitions returns every currently blocked directed link as "from->to",
// sorted.
//
// Sorted because this is built by walking a map, and an unsorted rendering would
// differ between runs. It is a diagnostic string, but diagnostics get pasted
// into assertions and trace lines, and a diagnostic that changes shape run to
// run is one more way to lose determinism.
func (nw *Network) Partitions() []string {
	out := make([]string, 0, len(nw.blocked))
	for l := range nw.blocked {
		out = append(out, l.from+"->"+l.to)
	}
	sort.Strings(out)
	return out
}

func (nw *Network) String() string {
	return fmt.Sprintf("netsim{nodes=%v blocked=%v %s}", nw.Nodes(), nw.Partitions(), nw.stats)
}

// renderGroup formats a group for the trace. It sorts a copy rather than the
// caller's slice: mutating an argument as a side effect of recording a trace
// line would be an unpleasant surprise, and if the caller reused that slice to
// drive its own loop, the reordering would change the run.
func renderGroup(g []string) string {
	cp := append([]string(nil), g...)
	sort.Strings(cp)
	return "[" + strings.Join(cp, " ") + "]"
}
