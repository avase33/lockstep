package netsim

import (
	"time"

	"github.com/avase33/lockstep/sim"
)

// Message is one delivery.
//
// It is a struct rather than a bare body because a receiver almost always needs
// the sender's name to reply, and threading that through the body type by hand
// is the first thing every user of a message-passing API ends up doing. The
// simulation-only fields (SentAt, Seq, Copy) cost nothing to carry and turn an
// unexplained failure into a one-line assertion.
type Message struct {
	// From and To are node names.
	From string
	To   string

	// Body is whatever the sender passed. The network never inspects it, never
	// copies it, and never formats it into the trace — see the package comment
	// on why a %v of a caller's struct is a determinism hazard.
	//
	// Because it is not copied, two nodes handed the same pointer share it.
	// That is faithful to a real system only if you would have serialised it;
	// if your protocol goes over the wire in production, send values, not
	// pointers, or the simulation will hide aliasing bugs that production will
	// not.
	Body any

	// SentAt is the virtual time Send was called. Carried so a receiver can
	// measure its own observed latency, and so a test can assert the one thing a
	// network must never do: deliver a message before it was sent.
	SentAt time.Time

	// Seq numbers messages network-wide in send order. Two copies of a
	// duplicated message share a Seq and differ in Copy.
	Seq uint64

	// Copy is 1 for the original and 2 for the duplicate the network
	// manufactured.
	//
	// This is a simulation affordance and a trap in equal measure: production
	// code cannot see it, so deduplicating on Copy is a test that proves
	// nothing. Use it to assert that duplication happened, never to survive it.
	Copy int
}

// Node is one endpoint on the network.
//
// A node is a handle, not a goroutine. You start the actor yourself with Sim.Go
// under the same name, which keeps the network out of the business of deciding
// how your node is structured and — more importantly — means Sim.Crash on that
// name kills the node and stops delivery to it in one move.
type Node struct {
	net   *Network
	name  string
	inbox *sim.Chan
}

// Name returns the node's name, which is also its actor name.
func (n *Node) Name() string { return n.name }

// Send queues a message to another node. It never blocks. Sending to yourself is
// allowed and behaves like any other link, faults included, because loopback in
// a real system is usually a different code path worth exercising.
func (n *Node) Send(to string, body any) { n.net.Send(n.name, to, body) }

// Recv blocks until a message arrives, and reports ok=false once the network is
// closed and this inbox is drained.
//
// The ok flag is what lets a node's loop be written the way a real one is:
//
//	for {
//		m, ok := node.Recv()
//		if !ok {
//			return // network shut down
//		}
//		switch v := m.Body.(type) {
//		case Ping:
//			node.Send(m.From, Pong{v.ID})
//		}
//	}
//
// Blocking here is safe in a way that blocking on a Go channel is not: the actor
// parks inside the scheduler, which knows it is waiting and will report it by
// name if the run deadlocks.
func (n *Node) Recv() (Message, bool) {
	v, ok := n.inbox.Recv()
	if !ok {
		return Message{}, false
	}
	return v.(Message), true
}

// TryRecv takes a message if one is already waiting, without blocking.
//
// Use it in a loop that has other work to do between messages. Note the
// difference from Recv: ok=false here means "nothing right now", not "the
// network is closed", so a bare TryRecv loop with no Sleep is a spin that
// advances virtual time not at all and will exhaust the step budget.
func (n *Node) TryRecv() (Message, bool) {
	v, ok := n.inbox.TryRecv()
	if !ok {
		return Message{}, false
	}
	return v.(Message), true
}

// Inbox exposes the underlying channel so a node can wait on messages and a
// timeout together:
//
//	i, v, ok := sim.Select(s, node.Inbox(), s.Clock().After(electionTimeout))
//
// That pattern is the whole reason this accessor exists. Every consensus
// implementation ever written needs "a message, or a timeout, whichever comes
// first", and without access to the raw Chan the only way to express it would be
// polling — which quantises virtual time and destroys the ability to place a
// message one nanosecond either side of a deadline.
//
// Values received from it are Message, boxed in any.
func (n *Node) Inbox() *sim.Chan { return n.inbox }

// Pending returns how many delivered messages are waiting to be read. Useful for
// asserting backpressure, and for spotting a receiver that has fallen far enough
// behind that the network is about to start dropping on overflow.
func (n *Node) Pending() int { return n.inbox.Len() }
