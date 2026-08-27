"""Read a self-directed run's journal and say what the model chose to study.

  python -m selfdirect.report --out saved/selfdirect_98m

Prints the curriculum the director settled on and what it bought, and writes
curriculum.png: the mixture over rounds above the per-arm probe loss it was
chasing.
"""
import argparse
import json
import os

from selfdirect.loop import JOURNAL_FILE


def read_journal(out_dir):
    path = os.path.join(out_dir, JOURNAL_FILE)
    with open(path) as f:
        rounds = [json.loads(line) for line in f if line.strip()]
    if not rounds:
        raise SystemExit(f"{path} is empty — has the run produced a round yet?")
    return rounds


def summarize(rounds):
    """Per-arm rows plus the run totals, from the journal alone."""
    names = list(rounds[0]['probs'])
    studied = {n: sum(r['studied'] == n for r in rounds) for n in names}
    start = rounds[0]['probe_before']
    now = rounds[-1]['probe_after']
    best = {n: min(r['probe_after'][n] for r in rounds) for n in names}
    final = rounds[-1]['probs']
    # Mean reward while studying each arm: what the director was reading. If
    # these are all the same, there was nothing for it to find.
    paid = {n: [r['reward'] for r in rounds if r['studied'] == n] for n in names}
    rows = [{'arm': n, 'rounds': studied[n], 'share': studied[n] / len(rounds),
             'weight': final[n], 'start': start[n], 'now': now[n],
             'best': best[n], 'forgotten': now[n] - best[n],
             'reward': sum(paid[n]) / len(paid[n]) if paid[n] else None}
            for n in names]
    return sorted(rows, key=lambda r: -r['weight'])


def mean_probe(probe):
    return sum(probe.values()) / len(probe)


def print_summary(rounds, rows):
    print(f"{len(rounds)} rounds, {rounds[-1]['step']} optimizer steps, "
          f"{sum(r['seconds'] for r in rounds) / 60:.1f} min")
    print(f"\n{'arm':<16}{'studied':>9}{'weight':>9}{'mean reward':>13}"
          f"{'probe start':>13}{'probe now':>11}{'best':>9}{'forgotten':>11}")
    for r in rows:
        reward = f"{r['reward']:+.5f}" if r['reward'] is not None else '-'
        print(f"{r['arm']:<16}{r['rounds']:>4} ({r['share']*100:2.0f}%)"
              f"{r['weight']*100:>8.1f}%{reward:>13}"
              f"{r['start']:>13.4f}{r['now']:>11.4f}"
              f"{r['best']:>9.4f}{r['forgotten']:>+11.4f}")
    start = mean_probe(rounds[0]['probe_before'])
    now = mean_probe(rounds[-1]['probe_after'])
    print(f"\nmean probe loss {start:.4f} -> {now:.4f} ({now - start:+.4f})")
    top, uniform = rows[0], 1 / len(rows)
    if top['weight'] < 1.25 * uniform:
        print("the director stayed near uniform — no arm paid consistently better")
    else:
        print(f"the director settled on {top['arm']} at {top['weight']*100:.0f}% "
              f"(uniform would be {uniform*100:.0f}%)")


def print_comparison(runs):
    """Head to head across runs that consumed the same tokens per round, so a
    round index is a fair x-axis for all of them."""
    w = max(len(label) for label, _ in runs) + 2
    print(f"\n{'run':<{w}}{'rounds':>7}{'probe start':>13}{'probe end':>11}"
          f"{'improved':>11}{'forgotten':>11}")
    for label, rounds in runs:
        start, end = mean_probe(rounds[0]['probe_before']), mean_probe(rounds[-1]['probe_after'])
        print(f"{label:<{w}}{len(rounds):>7}{start:>13.4f}{end:>11.4f}"
              f"{start - end:>+11.4f}{rounds[-1]['forgetting']:>11.4f}")
    best = min(runs, key=lambda r: mean_probe(r[1][-1]['probe_after']))
    print(f"lowest final probe loss: {best[0]}")


def plot_comparison(runs, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for label, rounds in runs:
        plt.plot([0] + [r['round'] for r in rounds],
                 [mean_probe(rounds[0]['probe_before'])] +
                 [mean_probe(r['probe_after']) for r in rounds],
                 label=label, linewidth=1.6)
    plt.xlabel('round (identical tokens per round across runs)')
    plt.ylabel('mean probe loss over all arms')
    plt.title('Self-directed curriculum vs. the mixture it was given')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"wrote {path}")


def plot(rounds, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(rounds[0]['probs'])
    steps = [r['round'] for r in rounds]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    top.stackplot(steps, *[[r['probs'][n] for r in rounds] for n in names],
                  labels=names, alpha=0.85)
    top.set_ylim(0, 1)
    top.set_xlim(0, steps[-1])
    top.set_ylabel('director weight')
    top.set_title('What the model chose to study')
    top.legend(loc='upper center', ncol=len(names), fontsize=8, framealpha=0.9)

    for n in names:
        bot.plot([0] + steps, [rounds[0]['probe_before'][n]] +
                 [r['probe_after'][n] for r in rounds], label=n, linewidth=1.4)
    bot.set_xlabel('round')
    bot.set_ylabel('probe loss')
    bot.set_title('The signal it was reading')
    bot.grid(True, alpha=0.3)
    bot.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='saved/selfdirect')
    parser.add_argument('--compare', nargs='*', default=[],
                        help='Other run directories to put head to head with --out')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    rounds = read_journal(args.out)
    print_summary(rounds, summarize(rounds))
    if not args.no_plot:
        plot(rounds, os.path.join(args.out, 'curriculum.png'))
    if args.compare:
        runs = [(os.path.basename(d.rstrip('/')), read_journal(d))
                for d in [args.out] + args.compare]
        print_comparison(runs)
        if not args.no_plot:
            plot_comparison(runs, os.path.join(args.out, 'comparison.png'))


if __name__ == '__main__':
    main()
