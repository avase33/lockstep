package linz

import "sort"

// This file is the checker proper: the Wing & Gong search, the memoisation that
// makes it usable, and the bookkeeping that lets a failure explain itself.
//
// # The algorithm in one paragraph
//
// Lay every invocation and every response out on a doubly linked list in time
// order. Walk forward from the head trying to linearise operations: for each
// invocation encountered, ask the model whether that operation could have taken
// effect next; if it could, remove both its invocation and its response from the
// list, advance the model's state, and start again from the head. If instead the
// walk reaches a RESPONSE whose operation is still in the list, stop — every
// operation still ahead was invoked after that response, so it cannot be next,
// and every operation before it has just been tried and refused. That is a dead
// end; undo the most recent choice and try the next candidate. The list empties
// exactly when a complete linearisation has been found.
//
// # Why the linked list
//
// Because "remove an operation and restart from the head" is the inner loop, and
// it has to be O(1). The list also encodes the real-time constraint for free:
// the only reason the walk ever stops early is a response entry, and a response
// entry is precisely a real-time barrier. There is no separate check for
// "does this ordering respect real time" anywhere in this package — the data
// structure is the check.
//
// The lift/unlift pair is the classic trick that makes it work: removing a node
// updates its neighbours but leaves the node's own prev/next pointers intact, so
// putting it back is four assignments and needs no extra storage. It is only
// correct because removals are undone in exactly the reverse order they were
// made, which the depth-first search guarantees.

// nodeKind distinguishes the two kinds of entry on the timeline.
type nodeKind uint8

const (
	kindCall nodeKind = iota
	kindReturn
)

// node is one entry — an invocation or a response — on the timeline list.
type node struct {
	kind  nodeKind
	value any // the input for an invocation, the output for a response
	id    int // index of the operation within its partition, in invocation order
	time  int64

	prev, next *node
	// match links an invocation to its own response. Only set on invocations,
	// which is also how the search tells the two kinds apart cheaply.
	match *node
}

// lift removes an invocation and its response from the list.
//
// Note what it does not do: it never writes to n.prev, n.next, match.prev or
// match.next. Those stale pointers are the saved undo record. (The one exception
// is an adjacent pair, where removing the invocation rewrites its response's
// prev; unlift restores that by putting the response back first and the
// invocation second, in that order, which is why the order below is not
// arbitrary.)
func (n *node) lift() {
	n.prev.next = n.next
	n.next.prev = n.prev
	m := n.match
	m.prev.next = m.next
	m.next.prev = m.prev
}

// unlift puts an invocation and its response back where they were.
func (n *node) unlift() {
	m := n.match
	m.prev.next = m
	m.next.prev = m
	n.prev.next = n
	n.next.prev = n
}

// bitset records which operations of a partition have been linearised so far.
// One bit per operation, indexed by the operation's position in invocation
// order. Setting and clearing are O(1); the only part that is ever copied is the
// window described below.
type bitset []uint64

func newBitset(n int) bitset { return make(bitset, (n+63)/64) }

func (b bitset) set(i int)   { b[i>>6] |= 1 << uint(i&63) }
func (b bitset) clear(i int) { b[i>>6] &^= 1 << uint(i&63) }

// ---------------------------------------------------------------------------
// Memoisation
// ---------------------------------------------------------------------------

// The memo table is what turns a factorial search into a feasible one. Two
// paths that have linearised the same SET of operations and reached the same
// model state face an identical remaining problem, so the second one can stop
// immediately. Recognising that requires a key, and the key has to be cheap,
// because it is built on every single search step.
//
// The obvious key — a bit per operation in the partition, cloned on every step —
// is O(n) to copy, so it makes the whole check O(n²). That is not hypothetical:
// this package did exactly that first, and measured 14 ms to check a
// 10,000-operation sequential history against 708 ms for a 100,000-operation
// one. Ten times the history, fifty times the cost, nearly all of it memcpy.
// With the window key below the same two histories cost 7.4 ms and 53 ms — seven
// times the cost for ten times the history, which is the linear scaling the
// package documentation claims.
//
// The trick is that the set is never arbitrary. Operations are numbered in
// invocation order, and an operation can only be linearised while it is in
// flight, so the set always looks like "everything before some point, plus a
// scattering within the handful of operations that overlap it". Write `base` for
// the lowest-numbered operation not yet linearised — the head of the list, which
// is O(1) to read. Then:
//
//   - every operation below base is linearised, by definition of base;
//   - every operation above the highest linearised one is not;
//   - only the span between them varies, and that span is the concurrency
//     width.
//
// So the key is base plus the words of the bitset covering that span: one or two
// words for any realistic workload, and one word for a history with no
// concurrency at all. Copying is O(width) instead of O(n), which is what makes
// cost linear in history length. (A history whose very first operation never
// returned can still stretch the span; correctness does not depend on the span
// being short, only speed does.)

// cacheEntry is one visited (linearised set, state) pair.
type cacheEntry struct {
	base  int
	words []uint64
	state State
}

// windowOf returns the span of the bitset that can vary, given base and the
// highest linearised operation. Bits below base are all set and bits above
// maxLin are all clear, so neither end carries information — but including whole
// words costs nothing and avoids any shifting.
func windowOf(b bitset, base, maxLin int) []uint64 {
	lo := base >> 6
	hi := 0
	if maxLin >= 0 {
		hi = (maxLin >> 6) + 1
	}
	if lo > len(b) {
		lo = len(b)
	}
	if hi > len(b) {
		hi = len(b)
	}
	if hi < lo {
		hi = lo
	}
	return b[lo:hi]
}

// hashWindow only has to spread well enough to keep buckets short; correctness
// comes from the comparison inside the bucket, so a collision costs time and
// never an answer.
func hashWindow(base int, words []uint64) uint64 {
	h := uint64(14695981039346656037)
	h = (h ^ uint64(base)) * 1099511628211
	for _, w := range words {
		h = (h ^ w) * 1099511628211
	}
	return h
}

func sameWindow(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// frame is one entry on the search's undo stack.
type frame struct {
	n     *node
	state State
	// prevMax restores the highest-linearised-operation counter on backtracking.
	// Storing it here rather than recomputing it keeps undo at O(1); the stack is
	// popped in exactly the order it was pushed, so the saved value is always the
	// right one.
	prevMax int
}

// searchStatus is the outcome of searching one partition.
type searchStatus uint8

const (
	searchOK searchStatus = iota
	searchFailed
	searchBudget
)

// searcher runs the algorithm over the operations of a single partition.
type searcher struct {
	model Model
	// ops is this partition's operations sorted by invocation time. The sort is
	// load-bearing, not cosmetic: the memo key above relies on operation numbers
	// following invocation order, so the searcher establishes that itself rather
	// than trusting the caller to have done it.
	ops  []Operation
	head *node
	tail *node
	n    int
	// width is the greatest number of operations in flight at once, computed
	// while the timeline is being built because the entries are already sorted by
	// then. It is reported to the caller: it is the variable the search cost is
	// exponential in, so it is the number that explains a slow or undecided
	// check.
	width int
	// complete is the number of operations that actually returned. Pending ones
	// are counted separately because they may be discarded; see run().
	complete int
	budget   *budgetState

	// Deepest dead end seen so far, kept for the failure report. "Deepest" is
	// the right thing to report: it is the longest prefix of a valid explanation
	// that exists at all, so the operations that could not extend it are where
	// the system's behaviour first became impossible.
	haveDeadEnd bool
	deadDepth   int
	deadOrder   []int
	deadState   State
	deadBlocked []blockedOp
	deadBarrier int // op id of the response that ended the walk, or -1
}

// blockedOp is one candidate that could not be linearised at the dead end, and
// why not.
type blockedOp struct {
	id int
	// rejected is true when the model said this output was impossible in this
	// state, and false when the branch was pruned because an equivalent
	// (set, state) pair had already been explored and led nowhere.
	rejected bool
}

func newSearcher(m Model, ops []Operation, b *budgetState) *searcher {
	sorted := make([]Operation, len(ops))
	copy(sorted, ops)
	// Callers hand over operations that are already in invocation order, and
	// checking that is a linear scan while sorting is not. On a short history the
	// sorts are most of the cost of the whole check, so the scan is worth it.
	byCall := func(i, j int) bool { return sorted[i].Call < sorted[j].Call }
	if !sort.SliceIsSorted(sorted, byCall) {
		sort.SliceStable(sorted, byCall)
	}

	s := &searcher{
		model:       m,
		ops:         sorted,
		n:           len(sorted),
		budget:      b,
		deadBarrier: -1,
	}

	// One slab for every node rather than 2n separate allocations. The nodes
	// live exactly as long as the searcher does, so there is nothing to be
	// gained from allocating them individually and two allocations per operation
	// to be lost.
	slab := make([]node, 2*len(sorted))
	entries := make([]*node, 0, 2*len(sorted))
	for i, op := range sorted {
		call, ret := &slab[2*i], &slab[2*i+1]
		*call = node{kind: kindCall, value: op.Input, id: i, time: op.Call}
		*ret = node{kind: kindReturn, value: op.observedOutput(), id: i, time: op.Return}
		if op.Pending {
			// A response later than every real one. This is how "the operation
			// may have taken effect at any point after it was invoked, including
			// after everything else finished" is expressed: with no response to
			// act as a barrier, nothing can force this operation to come earlier.
			ret.time = pendingReturn
		} else {
			s.complete++
		}
		call.match = ret
		entries = append(entries, call, ret)
	}

	// Sort by time; at equal times put invocations before responses, so that two
	// events sharing a timestamp are treated as concurrent rather than ordered.
	// The id tiebreak is not cosmetic: without it the order of same-timestamp
	// entries would depend on the sort implementation, and this package would
	// hand back different failure reports for the same history on different Go
	// versions.
	sort.Slice(entries, func(i, j int) bool {
		a, b := entries[i], entries[j]
		if a.time != b.time {
			return a.time < b.time
		}
		if a.kind != b.kind {
			return a.kind == kindCall
		}
		return a.id < b.id
	})

	// Sentinels at both ends so lift and unlift never test for nil.
	s.head = &node{id: -1}
	s.tail = &node{id: -1}
	prev := s.head
	inFlight := 0
	for _, e := range entries {
		prev.next = e
		e.prev = prev
		prev = e
		// The entries are in time order here, so the same walk that links them
		// measures the concurrency width for free.
		if e.kind == kindCall {
			inFlight++
			if inFlight > s.width {
				s.width = inFlight
			}
		} else {
			inFlight--
		}
	}
	prev.next = s.tail
	s.tail.prev = prev
	return s
}

// run searches for a linearisation of this partition.
//
// # Complexity, honestly
//
// The search space is every order in which the still-eligible operations could
// have taken effect, so the naive bound is factorial. Two things cut it down.
//
// Real time does most of the work: an operation can only be linearised while it
// is in flight, so the branching factor at each step is not the number of
// remaining operations but the number of operations whose intervals currently
// overlap. Call that the concurrency width, w.
//
// Memoisation does the rest, collapsing the tree of orderings into a graph of
// reachable (set, state) pairs. The number of reachable sets is bounded by the
// ways to choose a subset of what is in flight, so the cost is roughly
// O(n · 2^w) model steps rather than O(n!).
//
// The consequence to remember: cost grows LINEARLY in history length and
// EXPONENTIALLY in how many operations overlap. A hundred-thousand-operation
// history with two clients is fine. A two-hundred-operation history with thirty
// clients all touching the same key may not finish this year. Partition by key,
// keep client counts per key modest, and set a budget. The measured numbers
// behind those claims are in the package documentation.
func (s *searcher) run() searchStatus {
	if s.n == 0 {
		return searchOK
	}

	state := s.model.Init()
	linearized := newBitset(s.n)
	cache := make(map[uint64][]cacheEntry)
	calls := make([]frame, 0, s.n)
	remaining := s.complete
	maxLin := -1

	entry := s.head.next
	for s.head.next != s.tail {
		// Every operation that actually returned has been placed, so whatever is
		// left never returned and may be discarded: a client that crashed is
		// allowed to have made no difference, and nothing that observed it is
		// left to contradict that. Stopping here is not an optimisation, it is
		// the second half of what a pending operation means.
		if remaining == 0 {
			return searchOK
		}
		if s.budget.tick() {
			return searchBudget
		}

		if entry != s.tail && entry.kind == kindCall {
			ok, next := s.model.Step(state, entry.value, entry.match.value)
			if ok {
				// Speculatively take the step so that the memo key can be read
				// straight off the live structures — the list head gives base and
				// the bitset gives the window. Undoing it is eight pointer writes,
				// which is cheaper than computing the same key indirectly.
				entry.lift()
				linearized.set(entry.id)
				newMax := maxLin
				if entry.id > newMax {
					newMax = entry.id
				}
				base := s.n
				if s.head.next != s.tail {
					base = s.head.next.id
				}
				window := windowOf(linearized, base, newMax)

				if !s.seen(cache, base, window, next) {
					s.remember(cache, base, window, next)
					calls = append(calls, frame{n: entry, state: state, prevMax: maxLin})
					state = next
					maxLin = newMax
					if !s.ops[entry.id].Pending {
						remaining--
					}
					entry = s.head.next
					continue
				}

				linearized.clear(entry.id)
				entry.unlift()
			}
			entry = entry.next
			continue
		}

		// A response entry (or, defensively, the end of the list): a real-time
		// barrier. Nothing further along may be linearised next, and everything
		// before it has just been refused. Back out the most recent choice.
		s.noteDeadEnd(calls, state, entry)
		if len(calls) == 0 {
			return searchFailed
		}
		top := calls[len(calls)-1]
		calls = calls[:len(calls)-1]
		state = top.state
		maxLin = top.prevMax
		linearized.clear(top.n.id)
		if !s.ops[top.n.id].Pending {
			remaining++
		}
		top.n.unlift()
		entry = top.n.next
	}
	return searchOK
}

func (s *searcher) seen(cache map[uint64][]cacheEntry, base int, window []uint64, state State) bool {
	for _, e := range cache[hashWindow(base, window)] {
		if e.base == base && sameWindow(e.words, window) && s.model.Equal(e.state, state) {
			return true
		}
	}
	return false
}

// remember copies the window, because the live bitset keeps changing underneath
// as the search advances and backtracks. Lookups above deliberately do not copy,
// so the hot path allocates only when it finds something new.
func (s *searcher) remember(cache map[uint64][]cacheEntry, base int, window []uint64, state State) {
	stored := make([]uint64, len(window))
	copy(stored, window)
	h := hashWindow(base, window)
	cache[h] = append(cache[h], cacheEntry{base: base, words: stored, state: state})
}

// noteDeadEnd records the deepest point the search ever reached, for the report.
//
// The recomputation below (asking the model again about each candidate) looks
// wasteful inside a hot loop, and would be — except that it only runs when a new
// deepest dead end is found, which happens at most n times per partition. What it
// buys is the difference between "not linearizable" and "after these seven
// operations the store held 1, and the read that returned 0 could not have
// happened there" — which is the difference between a report that helps and one
// that does not.
func (s *searcher) noteDeadEnd(calls []frame, state State, barrier *node) {
	if s.haveDeadEnd && len(calls) <= s.deadDepth {
		return
	}
	s.haveDeadEnd = true
	s.deadDepth = len(calls)

	s.deadOrder = s.deadOrder[:0]
	for _, f := range calls {
		s.deadOrder = append(s.deadOrder, f.n.id)
	}
	s.deadState = state

	s.deadBlocked = s.deadBlocked[:0]
	for cur := s.head.next; cur != barrier && cur != s.tail; cur = cur.next {
		if cur.kind != kindCall {
			continue
		}
		ok, _ := s.model.Step(state, cur.value, cur.match.value)
		s.deadBlocked = append(s.deadBlocked, blockedOp{id: cur.id, rejected: !ok})
	}

	s.deadBarrier = -1
	if barrier != nil && barrier != s.tail {
		s.deadBarrier = barrier.id
	}
}
