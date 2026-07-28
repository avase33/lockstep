package linz

import (
	"fmt"
	"maps"
	"reflect"
	"sort"
	"strings"
)

// State is whatever a Model uses to represent the value of the object it
// specifies: an int for a register, a map for a key-value store, a struct for
// anything richer.
//
// It is an alias for any rather than a defined type, so nothing has to be
// converted at the boundary. The name exists purely to make the Model interface
// readable — a signature that says (State, any, any) tells you which argument is
// which, and (any, any, any) does not.
//
// One rule governs everything in this package: a State must be treated as
// IMMUTABLE. The checker keeps old states in a memoisation table and returns to
// them when it backtracks, so a Step that mutates the state it was handed
// corrupts entries the search has already recorded, and the result is a verdict
// that depends on the search order. If your state is a map or a slice, copy it
// before you change it. NewKVModel does exactly that, and the copy is cheap
// precisely because partitioning keeps each state tiny.
type State = any

// Model is a sequential specification: the single-threaded, obviously-correct
// implementation that the concurrent system under test claims to behave like.
//
// The checker never runs your real system. It replays the recorded history
// against this specification, one operation at a time, in orders of its own
// choosing, looking for one order that reproduces every observed output. So the
// Model must be a pure function of (state, input) — no clocks, no globals, no
// randomness. A Model that answers differently on the second call will make the
// checker report violations that are artefacts of the Model, which is the most
// expensive kind of false alarm because the bug hunt starts in the wrong place.
//
// # Why Step takes the output
//
// A specification that only computed the next state could tell you whether an
// operation was legal, but not whether the system returned the right answer —
// and returning the wrong answer is the entire failure mode linearizability
// exists to catch. Step is therefore a predicate over a whole (state, input,
// output) triple: "could an object in this state, given this input, have
// returned this output, and if so what state is it in afterwards?"
//
// # Why Equal exists
//
// Memoisation. The search reaches the same (set of already-ordered operations,
// resulting state) pair by many different routes, and pruning the repeats is
// what keeps the checker out of factorial time. To recognise a repeat, the
// checker must compare states, and only you know when two of your states are
// the same. Making Equal too strict is safe but slow — the search re-explores
// work it has already done. Making it too loose is not safe: the checker will
// prune a branch that was genuinely different and can report a linearizable
// history as a violation.
//
// # Operations that never returned
//
// When a client crashed mid-call, the checker still has to consider that the
// operation may have taken effect — see the package documentation for why. It
// passes NoResponse as the output in that case, meaning "any output this
// operation could legally have produced is acceptable; just tell me the state it
// leaves behind". A Model that does not recognise NoResponse will refuse every
// such placement and can report a perfectly good history as a violation.
//
// The reliable way to avoid that mistake is to not write Step at all: implement
// Deterministic instead and let FromDeterministic handle NoResponse for you.
// Both built-in models are built that way.
//
// # What this interface cannot express, and what to do about it
//
// Step returns ONE next state, so the state after an operation must be a
// function of the state before it, the input, and the output that was observed.
//
// Nondeterminism in the output is fine: Step is a predicate, so the same state
// and input may legally accept several different outputs, and the next state may
// depend on which one actually came back. A "read may return any value written
// within the last second" specification fits comfortably.
//
// What does not fit is hidden nondeterminism in the state — an operation that
// could leave the object in either of two states without the difference showing
// up in its output. There is no way to say that here, and no linearizability
// checker of this shape can search it directly. The standard workaround is to
// make the state a SET of candidate states and have Step advance all of them at
// once, discarding the candidates that contradict the observed output; Equal
// then compares sets. That works, and it costs whatever the branching costs.
type Model interface {
	// Init returns the state of a freshly created object, before any operation
	// in the history has run.
	Init() State

	// Step reports whether an object in this state could have accepted this
	// input and produced this output, and returns the resulting state. The
	// returned state is only meaningful when ok is true.
	Step(state State, input, output any) (ok bool, next State)

	// Equal reports whether two states are interchangeable for every future
	// operation. See the note above on the cost of getting it wrong in each
	// direction.
	Equal(a, b State) bool
}

// Deterministic is the easier way to write a specification, and the one to
// reach for first.
//
// Most specifications are functions: given the state and the input there is
// exactly one legal output and exactly one next state. Saying so directly is
// less code than a Step predicate, and it removes two ways to be subtly wrong —
// forgetting to handle NoResponse for operations that never returned, and
// accidentally accepting an output the specification does not actually permit.
//
// Wrap one with FromDeterministic to get a Model.
//
// If more than one output is legal for the same state and input — a read that
// may return any recently written value, an iterator whose order is unspecified
// — this interface cannot say so, and you should implement Model directly:
// Model's Step is a predicate and can accept several different outputs. The
// limitation neither of them escapes is described under Model.
type Deterministic interface {
	// Init returns the state of a freshly created object.
	Init() State

	// Apply returns the output the specification requires for this input, and
	// the state afterwards. It must not modify the state it is given.
	Apply(state State, input any) (output any, next State)

	// Equal reports whether two states are interchangeable. See Model.Equal.
	Equal(a, b State) bool
}

// FromDeterministic adapts a Deterministic specification into a Model.
//
// The adapter is where NoResponse is handled: for an operation that never
// returned, it computes the state Apply would produce and accepts whatever
// output the specification would have given, because the client never saw one
// and therefore nothing in the history can contradict it. Getting that right in
// one place, once, is the reason this adapter exists.
//
// Optional extensions of the wrapped value — Partitioner, Describer — keep
// working through the adapter; the checker looks through the wrapper for them.
func FromDeterministic(d Deterministic) Model { return deterministicModel{d} }

type deterministicModel struct{ d Deterministic }

func (m deterministicModel) Init() State { return m.d.Init() }

func (m deterministicModel) Equal(a, b State) bool { return m.d.Equal(a, b) }

func (m deterministicModel) Step(state State, input, output any) (bool, State) {
	want, next := m.d.Apply(state, input)
	if IsNoResponse(output) {
		return true, next
	}
	return outputsEqual(want, output), next
}

func (m deterministicModel) unwrap() any { return m.d }

// outputsEqual compares a specified output with an observed one.
//
// reflect.DeepEqual rather than ==, because outputs are frequently structs
// containing slices or maps, and == panics at runtime on those rather than
// failing to compile. A checker that panics halfway through a CI run when
// somebody adds a field to their output type is not worth the nanoseconds saved.
func outputsEqual(a, b any) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return reflect.DeepEqual(a, b)
}

// noResponse is the type of the NoResponse sentinel. It is unexported and
// zero-sized so that no value a caller could plausibly record as a real output
// can ever be mistaken for it.
type noResponse struct{}

func (noResponse) String() string { return "<no response>" }

// NoResponse is the output the checker supplies for an operation that was
// invoked but never returned.
//
// It means "the client never learned what this operation returned, so no output
// can contradict the specification — decide only whether the operation could
// have run at all, and tell me the state it leaves behind".
//
// You only encounter this if you implement Model yourself; FromDeterministic
// deals with it. Handle it with IsNoResponse rather than ==, so that the
// representation stays free to change.
var NoResponse any = noResponse{}

// IsNoResponse reports whether an output is the NoResponse sentinel, i.e.
// whether the checker is asking about an operation that never returned.
func IsNoResponse(output any) bool {
	_, ok := output.(noResponse)
	return ok
}

// Partitioner is the optional Model extension that unlocks the single most
// valuable optimisation in this package.
//
// Linearizability is a *local* property: a history over several independent
// objects is linearizable exactly when the sub-history of each object is
// linearizable on its own. (This is Herlihy and Wing's compositionality result,
// and it is not true of most other consistency conditions — it is the main
// practical reason to prefer linearizability as a correctness criterion.)
//
// So a history of 1,000 operations over 100 independent keys is not one
// search of 1,000 operations. It is 100 searches of about 10, and because the
// search cost is exponential in how many operations overlap in time, that
// difference is the difference between microseconds and never finishing. On this
// package's own benchmarks, checking such a history partitioned takes 870 µs and
// checking it whole takes 505 ms; the full table is in the package
// documentation, and the ratio there ranges from 4x to 700x depending on how
// ambiguous the interleaving happens to be.
//
// Implement it whenever your operations touch independent objects, which for a
// storage system means: return the key.
type Partitioner interface {
	// PartitionKey returns which independent object this input touches.
	//
	// It must be a comparable value — it is used as a map key — and a string or
	// an int is almost always the right choice. Operations that share a key are
	// checked together; operations with different keys never interact.
	//
	// Returning the same key for operations that are NOT independent (a
	// multi-key transaction, a global counter read) is a correctness bug in the
	// other direction: the checker would then miss violations that only show up
	// when the two objects are considered together. If an operation spans
	// objects, the model cannot be partitioned at all.
	PartitionKey(input any) any
}

// Describer is the optional Model extension that makes failure reports readable.
//
// Without it a report renders inputs and outputs with %v, which for a struct
// gives you {1 x 7} and leaves the reader to guess. With it you get
// `put("x", "7")`. A checker's output is read exactly when someone is confused
// and under time pressure, so this is worth the ten lines it costs.
type Describer interface {
	// DescribeOperation renders one invocation and its response, e.g.
	// `read() -> 7`. The output is NoResponse for an operation that never
	// returned.
	DescribeOperation(input, output any) string
	// DescribeState renders a state, e.g. `7` or `{x: 1, y: 2}`.
	DescribeState(state State) string
}

// Unpartitioned returns a Model that behaves identically to m but hides its
// Partitioner, forcing the checker to treat the whole history as one problem.
//
// This exists for two reasons, both about trust. First, it makes "does
// partitioning change the answer?" a question you can actually ask — this
// package's own tests use it to assert that both paths agree, because a
// partitioner that silently splits operations that were not independent would
// otherwise turn missed bugs into green tests. Second, when you suspect your
// PartitionKey is wrong, it is the fastest way to find out.
//
// It is far slower on any history with more than a handful of keys. That is the
// point of partitioning.
func Unpartitioned(m Model) Model {
	d, _ := extension[Describer](m)
	return unpartitioned{Model: m, desc: d}
}

// unpartitioned deliberately does NOT implement unwrap: hiding the Partitioner
// is its entire job, and the checker finds extensions by unwrapping. It forwards
// Describer explicitly, because losing the failure report as a side effect of
// hiding the partitioner would be an unpleasant surprise.
type unpartitioned struct {
	Model
	desc Describer
}

func (u unpartitioned) DescribeOperation(input, output any) string {
	if u.desc == nil {
		return defaultDescribeOperation(input, output)
	}
	return u.desc.DescribeOperation(input, output)
}

func (u unpartitioned) DescribeState(state State) string {
	if u.desc == nil {
		return fmt.Sprintf("%v", state)
	}
	return u.desc.DescribeState(state)
}

// unwrapper is implemented by Models that wrap another value which may carry the
// optional extensions.
type unwrapper interface{ unwrap() any }

// extension finds an optional interface on a Model, looking through any
// adapters that wrap it.
//
// Without this, FromDeterministic would swallow the Partitioner of every model
// it wraps and quietly turn a millisecond check into an hour-long one — a
// performance cliff with no error message, which is the worst kind.
func extension[T any](m Model) (T, bool) {
	var zero T
	var cur any = m
	for cur != nil {
		if t, ok := cur.(T); ok {
			return t, true
		}
		u, ok := cur.(unwrapper)
		if !ok {
			break
		}
		cur = u.unwrap()
	}
	return zero, false
}

func defaultDescribeOperation(input, output any) string {
	if IsNoResponse(output) {
		return fmt.Sprintf("%v -> (never returned)", input)
	}
	return fmt.Sprintf("%v -> %v", input, output)
}

// ---------------------------------------------------------------------------
// Register model
// ---------------------------------------------------------------------------

// RegisterOp names the operations of the register model.
type RegisterOp uint8

const (
	// OpRead returns the current value and changes nothing.
	OpRead RegisterOp = iota
	// OpWrite replaces the current value unconditionally.
	OpWrite
	// OpCAS replaces the current value only if it matches an expected value,
	// and reports whether it did.
	OpCAS
)

func (o RegisterOp) String() string {
	switch o {
	case OpRead:
		return "read"
	case OpWrite:
		return "write"
	case OpCAS:
		return "cas"
	}
	return fmt.Sprintf("RegisterOp(%d)", uint8(o))
}

// RegisterInput is the input of one register operation. Build them with Read,
// Write and CAS rather than by hand; the constructors make call sites read like
// the operation they describe.
type RegisterInput struct {
	Op RegisterOp
	// Value is the value to write, or the new value for a compare-and-swap.
	Value int
	// Compare is the value a compare-and-swap expects to find. Unused otherwise.
	Compare int
}

func (in RegisterInput) String() string {
	switch in.Op {
	case OpRead:
		return "read()"
	case OpWrite:
		return fmt.Sprintf("write(%d)", in.Value)
	case OpCAS:
		return fmt.Sprintf("cas(%d -> %d)", in.Compare, in.Value)
	}
	return fmt.Sprintf("%v(%d)", in.Op, in.Value)
}

// Read builds the input of a register read. Its recorded output is the int the
// read returned.
func Read() RegisterInput { return RegisterInput{Op: OpRead} }

// Write builds the input of a register write. A write's response carries no
// information, so record its output as nil.
func Write(v int) RegisterInput { return RegisterInput{Op: OpWrite, Value: v} }

// CAS builds the input of a compare-and-swap. Its recorded output is a bool:
// true if the swap happened.
//
// Worth having in the box because compare-and-swap is where linearizability
// violations actually live. A read that returns a stale value is often
// tolerable; two clients both being told their compare-and-swap succeeded is
// never tolerable, and it is exactly the kind of bug a real-time-aware checker
// catches and an eyeball does not.
func CAS(old, new int) RegisterInput {
	return RegisterInput{Op: OpCAS, Compare: old, Value: new}
}

// NewRegisterModel returns the specification of a single shared integer
// supporting read, write and compare-and-swap.
//
// The register starts at 0. If the system under test starts somewhere else,
// record an initial write at the head of the history rather than reaching for a
// configuration knob — the history then says what happened, which is what you
// want to be reading at 3am.
//
// Inputs must be RegisterInput. Outputs must be: int for a read, nil for a
// write, bool for a compare-and-swap. Recording the wrong shape is reported as
// a violation rather than silently ignored, because a harness that logs
// something other than what the system returned makes every subsequent verdict
// meaningless.
//
// The register is a single object, so this model deliberately does NOT
// implement Partitioner. Every operation on it interacts with every other, and
// pretending otherwise would hide real violations.
func NewRegisterModel() Model { return FromDeterministic(registerModel{}) }

type registerModel struct{}

func (registerModel) Init() State { return 0 }

func (registerModel) Equal(a, b State) bool { return a.(int) == b.(int) }

func (registerModel) Apply(state State, input any) (any, State) {
	s := state.(int)
	in, ok := input.(RegisterInput)
	if !ok {
		panic(fmt.Sprintf("linz: register model got input of type %T, want linz.RegisterInput", input))
	}
	switch in.Op {
	case OpRead:
		return s, s
	case OpWrite:
		return nil, in.Value
	case OpCAS:
		if s == in.Compare {
			return true, in.Value
		}
		return false, s
	}
	panic(fmt.Sprintf("linz: register model got unknown operation %v", in.Op))
}

func (registerModel) DescribeOperation(input, output any) string {
	in := input.(RegisterInput)
	if IsNoResponse(output) {
		return fmt.Sprintf("%v (never returned)", in)
	}
	if in.Op == OpWrite {
		return fmt.Sprintf("%v -> ok", in)
	}
	return fmt.Sprintf("%v -> %v", in, output)
}

func (registerModel) DescribeState(state State) string { return fmt.Sprintf("%v", state) }

// ---------------------------------------------------------------------------
// Key-value model
// ---------------------------------------------------------------------------

// KVOp names the operations of the key-value model.
type KVOp uint8

const (
	// OpGet returns the value at a key, or "" if the key is absent.
	OpGet KVOp = iota
	// OpPut sets the value at a key.
	OpPut
	// OpDelete removes a key.
	OpDelete
)

func (o KVOp) String() string {
	switch o {
	case OpGet:
		return "get"
	case OpPut:
		return "put"
	case OpDelete:
		return "delete"
	}
	return fmt.Sprintf("KVOp(%d)", uint8(o))
}

// KVInput is the input of one key-value operation. Build them with Get, Put and
// Delete.
type KVInput struct {
	Op    KVOp
	Key   string
	Value string
}

func (in KVInput) String() string {
	switch in.Op {
	case OpGet:
		return fmt.Sprintf("get(%q)", in.Key)
	case OpPut:
		return fmt.Sprintf("put(%q, %q)", in.Key, in.Value)
	case OpDelete:
		return fmt.Sprintf("delete(%q)", in.Key)
	}
	return fmt.Sprintf("%v(%q)", in.Op, in.Key)
}

// Get builds the input of a read. Its recorded output is the string that was
// read, with "" meaning "key absent".
func Get(key string) KVInput { return KVInput{Op: OpGet, Key: key} }

// Put builds the input of a write. Record its output as nil.
func Put(key, value string) KVInput { return KVInput{Op: OpPut, Key: key, Value: value} }

// Delete builds the input of a removal. Record its output as nil.
func Delete(key string) KVInput { return KVInput{Op: OpDelete, Key: key} }

// NewKVModel returns the specification of a key-value store with get, put and
// delete, and — importantly — a Partitioner keyed on the key.
//
// This is the model to reach for when testing a replicated store, and the
// partitioning is what makes it usable on realistic histories: see Partitioner
// for the measured difference.
//
// A missing key and a key holding "" are the same thing here. That is a real
// limitation and it is deliberate: distinguishing them costs every state an
// extra allocation and buys almost nothing, since a system that confuses the two
// is broken in a way this checker is not the cheapest tool to find. If you need
// the distinction, copy this model and use a pointer or a struct as the value.
//
// The state is a map even though partitioning means it holds at most one key.
// Keeping it a map is what lets Unpartitioned check the very same model over a
// whole history, which is how the tests establish that partitioning does not
// change the verdict. The per-step copy that immutability requires is therefore
// a copy of a one-entry map, which is why this is not the bottleneck it looks
// like.
func NewKVModel() Model { return FromDeterministic(kvModel{}) }

type kvModel struct{}

func (kvModel) Init() State { return map[string]string{} }

func (kvModel) Equal(a, b State) bool {
	return maps.Equal(a.(map[string]string), b.(map[string]string))
}

func (kvModel) PartitionKey(input any) any {
	in, ok := input.(KVInput)
	if !ok {
		panic(fmt.Sprintf("linz: kv model got input of type %T, want linz.KVInput", input))
	}
	return in.Key
}

func (kvModel) Apply(state State, input any) (any, State) {
	s := state.(map[string]string)
	in, ok := input.(KVInput)
	if !ok {
		panic(fmt.Sprintf("linz: kv model got input of type %T, want linz.KVInput", input))
	}
	switch in.Op {
	case OpGet:
		return s[in.Key], s
	case OpPut:
		// Copy-on-write. The checker holds references to earlier states in its
		// memoisation table and returns to them on backtracking, so mutating in
		// place here would rewrite history the search has already decided about.
		next := maps.Clone(s)
		next[in.Key] = in.Value
		return nil, next
	case OpDelete:
		if _, present := s[in.Key]; !present {
			return nil, s
		}
		next := maps.Clone(s)
		delete(next, in.Key)
		return nil, next
	}
	panic(fmt.Sprintf("linz: kv model got unknown operation %v", in.Op))
}

func (kvModel) DescribeOperation(input, output any) string {
	in := input.(KVInput)
	if IsNoResponse(output) {
		return fmt.Sprintf("%v (never returned)", in)
	}
	if in.Op == OpGet {
		if s, ok := output.(string); ok && s == "" {
			return fmt.Sprintf("%v -> (absent)", in)
		}
		return fmt.Sprintf("%v -> %v", in, output)
	}
	return fmt.Sprintf("%v -> ok", in)
}

func (kvModel) DescribeState(state State) string {
	s, ok := state.(map[string]string)
	if !ok {
		return fmt.Sprintf("%v", state)
	}
	if len(s) == 0 {
		return "{}"
	}
	keys := make([]string, 0, len(s))
	for k := range s {
		keys = append(keys, k)
	}
	// Sorted, because an unsorted map walk would make the same failure print
	// differently on every run and turn a diffable report into noise.
	sort.Strings(keys)
	var b strings.Builder
	b.WriteByte('{')
	for i, k := range keys {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%s: %q", k, s[k])
	}
	b.WriteByte('}')
	return b.String()
}
