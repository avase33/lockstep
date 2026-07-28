// Command kvstore-demo sweeps seeds against the replicated key-value store in
// package kvstore and reports what lockstep found.
//
//	go run ./examples/kvstore/cmd
//
// The default run drives the buggy read path — reads answered by whichever
// single replica replies first — and it fails, which is the point. Every line
// of the report is derived from the run it describes, and the seed printed
// alongside a violation replays it exactly:
//
//	go run ./examples/kvstore/cmd -seed=0x00000001
//
// Add -story to see which replica held what, and when. The checker says a read
// returned a value no ordering allows; the story says why, and the two together
// are usually the whole diagnosis:
//
//	go run ./examples/kvstore/cmd -seed=0x00000001 -story
//
// To see the same workload against the corrected read path, which should never
// produce a violation however many seeds it is given:
//
//	go run ./examples/kvstore/cmd -mode=quorum -seeds=200
//
// The command exits non-zero when a violation is found. That is not the demo
// failing; it is the store failing, which is what a linearizability check in CI
// is supposed to do about a store like this one.
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/avase33/lockstep/examples/kvstore"
	"github.com/avase33/lockstep/linz"
	"github.com/avase33/lockstep/sim"
)

func main() {
	var (
		modeFlag  = flag.String("mode", "single", `read path under test: "single" (buggy) or "quorum" (fixed)`)
		seedFlag  = flag.String("seed", "", "run this one seed in detail instead of sweeping, e.g. -seed=0x00000001")
		fromFlag  = flag.String("from", "0x1", "first seed of the sweep")
		seedsFlag = flag.Int("seeds", 8, "how many seeds to sweep")
		storyFlag = flag.Bool("story", false, "on a violation, print the faults and replica writes that touched the failing key")
	)
	flag.Parse()

	mode, err := parseMode(*modeFlag)
	if err != nil {
		fail(err)
	}
	opt := kvstore.DefaultOptions(mode)

	if *seedFlag != "" {
		seed, err := parseSeed(*seedFlag)
		if err != nil {
			fail(err)
		}
		os.Exit(one(seed, opt, *storyFlag))
	}

	from, err := parseSeed(*fromFlag)
	if err != nil {
		fail(err)
	}
	os.Exit(sweep(from, *seedsFlag, opt, *storyFlag))
}

// sweep runs a range of seeds, printing one line each, and then reports the
// first violation in full.
//
// It runs the whole range rather than stopping at the first failure, because the
// interesting number is not "is there a bad seed" — for this store there always
// is — but how many of them are bad. A defect that shows up in most seeds is one
// a sweep will catch tonight; one that shows up in a twentieth is why the sweep
// has to be a sweep.
func sweep(from sim.Seed, count int, opt kvstore.Options, story bool) int {
	fmt.Printf("lockstep: sweeping %d seeds against kvstore (%s)\n", count, opt.Mode)
	describe(opt)
	fmt.Println()

	var first kvstore.Outcome
	violations, broke := 0, false
	// Wall clock, deliberately, and the only one in this example. It measures
	// how long the sweep took the person watching it, which is a fact about this
	// machine and not about the runs — every number inside a run comes from the
	// virtual clock.
	start := time.Now()

	kvstore.Sweep(from, count, opt, func(out kvstore.Outcome) bool {
		fmt.Printf("  seed %s  %s\n", out.Seed, out.Summary())
		if out.SimErr != nil {
			broke = true
			first = out
			return false
		}
		if out.Violated() {
			violations++
			if violations == 1 {
				first = out
			}
		}
		return true
	})

	fmt.Printf("\n%d of %d seeds checked in %s\n", count, count, time.Since(start).Round(time.Millisecond))

	if broke {
		fmt.Printf("\nBROKEN: the simulation itself failed on seed %s\n%v\n", first.Seed, first.SimErr)
		return 2
	}
	if violations == 0 {
		fmt.Printf("no linearizability violation found (%s)\n", opt.Mode)
		return 0
	}

	fmt.Printf("\nFAIL: linearizability violated in %d of %d seeds\n", violations, count)
	report(first, story)
	return 1
}

// one runs a single seed and prints everything about it. This is what a seed
// printed by a sweep is for.
func one(seed sim.Seed, opt kvstore.Options, story bool) int {
	fmt.Printf("lockstep: replaying seed %s against kvstore (%s)\n", seed, opt.Mode)
	describe(opt)
	fmt.Println()

	out := kvstore.Run(seed, opt)
	if out.SimErr != nil {
		fmt.Printf("BROKEN: the simulation itself failed\n%v\n", out.SimErr)
		return 2
	}

	fmt.Printf("  operations: %d returned, %d never returned, over %v\n",
		out.Returned, out.Pending, out.KeysTouched())
	fmt.Printf("  network:    %s\n", out.Net)
	fmt.Printf("  faults:     %s\n", strings.Join(out.Faults, "\n              "))
	fmt.Printf("  replicas:   %s\n", divergence(out))
	fmt.Printf("  simulation: %d scheduling decisions over %s of virtual time, trace %s\n",
		out.Steps, out.VirtualTime.Round(time.Millisecond), out.TraceHash)
	fmt.Println()

	switch out.Check.Status {
	case linz.NotLinearizable:
		fmt.Println("FAIL: linearizability violated")
		report(out, story)
		return 1
	case linz.Unknown:
		fmt.Printf("UNKNOWN: %v\n", out.Check)
		return 2
	default:
		fmt.Printf("PASS: %v\n", out.Check)
		return 0
	}
}

// report prints the failure the way somebody looking at CI needs it: the seed
// first, then what went wrong, then the command that brings it back.
func report(out kvstore.Outcome, story bool) {
	fmt.Printf("  seed:   %s\n", out.Seed)
	fmt.Printf("  trace:  %s\n", out.TraceHash)
	fmt.Printf("  faults: %s\n", strings.Join(out.Faults, "; "))
	fmt.Println()
	fmt.Println(indent(out.Check.String(), "  "))
	if story {
		key := out.FailingKey()
		fmt.Printf("  what happened to key %q, as the simulation recorded it:\n\n", key)
		fmt.Println(indent(strings.Join(out.Story(key), "\n"), "    "))
		fmt.Println()
	}
	fmt.Printf("  repro: %s\n", out.Repro())
}

func describe(opt kvstore.Options) {
	o := opt.Normalised()
	fmt.Printf("  cluster: %d replicas, quorum of %d, %s request timeout, %d retries\n",
		o.Cluster.Replicas, o.Cluster.Replicas/2+1, o.Cluster.Timeout, o.Cluster.Retries)
	fmt.Printf("  clients: %d, each doing %d operations over %d keys, %.0f%% writes\n",
		o.Clients, o.Ops, o.Keys, o.WriteRatio*100)
	fmt.Printf("  network: %s-%s latency, %.0f%% loss, %.0f%% duplication%s\n",
		o.Faults.MinLatency, o.Faults.MaxLatency, o.Faults.DropRate*100, o.Faults.DuplicateRate*100,
		faultSuffix(o))
}

func faultSuffix(o kvstore.Options) string {
	var extra []string
	if o.Partition {
		extra = append(extra, "one replica isolated then healed")
	}
	if o.CrashRestart {
		extra = append(extra, "one replica crashed then restarted")
	}
	if len(extra) == 0 {
		return ""
	}
	return ", " + strings.Join(extra, ", ")
}

func divergence(out kvstore.Outcome) string {
	if out.Diverged {
		return "held different values for the same key at some point in the run"
	}
	return "never disagreed about any key"
}

func parseMode(s string) (kvstore.Mode, error) {
	switch s {
	case "single":
		return kvstore.SingleReplicaReads, nil
	case "quorum":
		return kvstore.QuorumReads, nil
	}
	return 0, fmt.Errorf("unknown -mode %q: want \"single\" or \"quorum\"", s)
}

// parseSeed accepts the form a seed prints itself in, so a seed copied out of a
// failure report can be pasted straight back in.
func parseSeed(s string) (sim.Seed, error) {
	v, err := strconv.ParseUint(strings.TrimSpace(s), 0, 64)
	if err != nil {
		return 0, fmt.Errorf("bad seed %q: want a number, e.g. -seed=0x00000001", s)
	}
	return sim.Seed(v), nil
}

func indent(s, prefix string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	for i, l := range lines {
		if l != "" {
			lines[i] = prefix + l
		}
	}
	return strings.Join(lines, "\n")
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "kvstore-demo:", err)
	flag.Usage()
	os.Exit(2)
}
