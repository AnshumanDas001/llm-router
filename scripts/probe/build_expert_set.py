"""Build the expert band: questions that separate a model that thinks from
one that doesn't.

Why this exists. On every set we had, mid (gemini-3.5-flash, thinking off)
and frontier (the same model, thinking on) scored the same: 1.00 on the
116-question eval set and 96% vs 97% on 288 BIG-Bench Hard / MATH-500
questions. BBH was built for models that *could not* write out their
working; mid writes out its working in its answer, so chain-of-thought
tasks are solved either way. With no measured difference the routing
policy can never choose the frontier, and it never did.

What a thinking budget buys is not working-out but *search*: trying a
branch, finding it fails, backing up. So every family here needs either
search or a long exact computation where one slip is fatal:

  countdown       hit a target from six numbers -- a search over
                  expressions; a linear first attempt almost never lands
  knights_knaves  7-8 islanders, one statement each -- a case split over
                  128-256 worlds with exactly one consistent
  zebra           a 4x4 logic grid with only the clues needed for a
                  unique answer -- propagation plus backtracking
  shortest_path   a weighted graph built so the greedy route is a trap
  grid_paths      count monotone lattice paths around blocked points --
                  an exact DP where one wrong cell poisons the total
  aime            AIME 2024/2025, the standard reasoning benchmark

Every generated question is solved by brute force here, so each answer is
known and unique, and grading is automatic (see app/evaluation/graders.py).
Generated rather than scraped, the puzzles are also not in anyone's
training data, and difficulty is a parameter rather than a guess.

    ./venv/bin/python -m scripts.probe.build_expert_set [--per-family 12] [--seed 7]
"""
import argparse
import heapq
import itertools
import json
import random
import urllib.parse
import urllib.request

import numpy as np

from app.paths import EXPERT_QUERIES

ANSWER_LINE = "\n\nEnd your reply with a final line of the form: Answer: <answer>"


def _question(family: str, i: int, query: str, **grading) -> dict:
    """The question as a person would ask it, with the answer-format
    instruction kept apart: the classifier embeds `query` alone, and an
    instruction no user would type made every reference look alike."""
    query, _, fmt = query.partition("\n\nEnd your reply")
    return {"id": f"x_{family}_{i:02d}", "query": query,
            "answer_format": "\n\nEnd your reply" + fmt, "difficulty": "expert",
            "task_type": family, "source": "generated", **grading}


# --- countdown --------------------------------------------------------------

def _combine(a: set, b: set) -> set:
    out = set()
    for x in a:
        for y in b:
            out.add(x + y)
            out.add(x * y)
            if x > y:
                out.add(x - y)
            elif y > x:
                out.add(y - x)
            if y and x % y == 0:
                out.add(x // y)
            if x and y % x == 0:
                out.add(y // x)
    return out


def _countdown_values(numbers: list[int]) -> dict:
    """{bitmask of numbers used: every value reachable using exactly those}."""
    n = len(numbers)
    values = {1 << i: {numbers[i]} for i in range(n)}
    for mask in range(1, 1 << n):
        if mask in values:
            continue
        acc = set()
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub < other:                       # each split once
                acc |= _combine(values[sub], values[other])
            sub = (sub - 1) & mask
        values[mask] = acc
    return values


def countdown(rng: random.Random, i: int) -> dict:
    """Six numbers, a three-digit target that needs at least five of them --
    no two- or three-number shortcut exists, so a lucky first guess can't
    land it."""
    while True:
        numbers = rng.sample([25, 50, 75, 100], 2) + [rng.randint(1, 10) for _ in range(4)]
        rng.shuffle(numbers)
        values = _countdown_values(numbers)
        by_size = {}
        for mask, vals in values.items():
            by_size.setdefault(bin(mask).count("1"), set()).update(vals)
        easy = set().union(*(by_size.get(k, set()) for k in (1, 2, 3, 4)))
        candidates = [t for t in (by_size[5] | by_size[6]) - easy if 101 <= t <= 999]
        if candidates:
            target = rng.choice(candidates)
            break
    query = (f"Using the numbers {', '.join(map(str, numbers))}, make exactly {target}. "
             "You may use +, -, * and /, and parentheses. Each number may be used at most "
             "once, and every intermediate result must be a positive whole number. "
             "Give the full expression on one line." + ANSWER_LINE.replace("<answer>", "<expression>"))
    return _question("countdown", i, query, eval_method="countdown", numbers=numbers, target=target)


# --- knights and knaves -----------------------------------------------------

NAMES = ["Ada", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lena"]


def _kk_statement(rng, speaker, people, n):
    others = [p for p in range(n) if p != speaker]
    a, b = rng.sample(others, 2)
    kind = rng.choice(["is", "both", "either", "xor", "iff", "count", "same"])
    if kind == "is":
        role = rng.choice([True, False])
        return (lambda w, a=a, role=role: w[a] == role,
                f"{people[a]} is a {'knight' if role else 'knave'}.")
    if kind == "both":
        return (lambda w, a=a, b=b: w[a] and w[b], f"{people[a]} and {people[b]} are both knights.")
    if kind == "either":
        return (lambda w, a=a, b=b: (not w[a]) or (not w[b]),
                f"At least one of {people[a]} and {people[b]} is a knave.")
    if kind == "xor":
        return (lambda w, a=a, b=b: w[a] != w[b],
                f"Exactly one of {people[a]} and {people[b]} is a knight.")
    if kind == "iff":
        return (lambda w, a=a, b=b: w[a] == (not w[b]),
                f"{people[a]} is a knight if and only if {people[b]} is a knave.")
    if kind == "same":
        return (lambda w, a=a, b=b: w[a] == w[b],
                f"{people[a]} and {people[b]} are the same kind.")
    k = rng.randint(1, n - 1)
    return (lambda w, k=k: sum(w) == k, f"Exactly {k} of the {n} of us are knights.")


def knights_knaves(rng: random.Random, i: int) -> dict:
    n = 7 if i % 2 == 0 else 8
    while True:
        people = rng.sample(NAMES, n)
        statements = [_kk_statement(rng, s, people, n) for s in range(n)]
        worlds = [w for w in itertools.product([True, False], repeat=n)
                  if all(w[s] == statements[s][0](w) for s in range(n))]
        if len(worlds) == 1:
            break
    lines = "\n".join(f"{people[s]} says: \"{statements[s][1]}\"" for s in range(n))
    knights = [people[p] for p in range(n) if worlds[0][p]]
    query = (f"On an island, every inhabitant is either a knight, who always tells the truth, "
             f"or a knave, who always lies. You meet {n} inhabitants: {', '.join(people)}.\n\n"
             f"{lines}\n\nExactly one assignment is consistent with these statements. "
             "Which of them are knights? List every knight, separated by commas "
             "(or 'none')." + ANSWER_LINE.replace("<answer>", "<names>"))
    return _question("knights_knaves", i, query, eval_method="answer_set", expected=knights)


# --- zebra ------------------------------------------------------------------

ZEBRA_ATTRS = {
    "name": ["Alice", "Bruno", "Chen", "Dara"],
    "drink": ["coffee", "tea", "milk", "juice"],
    "pet": ["cat", "dog", "parrot", "fish"],
    "job": ["baker", "doctor", "pilot", "teacher"],
}
ZEBRA_PHRASE = {
    "name": lambda v: v, "drink": lambda v: f"the {v} drinker",
    "pet": lambda v: f"the {v} owner", "job": lambda v: f"the {v}",
}


def _zebra_space() -> np.ndarray:
    """Every assignment as positions: pos[world, attribute, item] = house."""
    perms = np.array(list(itertools.permutations(range(4))))       # 24 x 4
    idx = np.array(list(itertools.product(range(24), repeat=4)))   # 331,776 x 4
    return perms[idx]                                              # W x 4 x 4


def zebra(rng: random.Random, i: int, space=None) -> dict:
    attrs = list(ZEBRA_ATTRS)
    space = _zebra_space() if space is None else space
    truth = space[rng.randrange(len(space))]

    def phrase(a, v):
        return ZEBRA_PHRASE[attrs[a]](ZEBRA_ATTRS[attrs[a]][v])

    def clue():
        a, b = rng.sample(range(4), 2)
        u, v = rng.randrange(4), rng.randrange(4)
        kind = rng.choice(["same", "same", "not_same", "left", "next", "before", "at", "not_at"])
        P, T = (lambda x, y: space[:, x, y]), (lambda x, y: truth[x, y])
        if kind == "same":
            v = int(np.where(truth[b] == truth[a, u])[0][0])
            return P(a, u) == P(b, v), f"{phrase(a, u).capitalize()} is {phrase(b, v)}."
        if kind == "not_same" and T(a, u) != T(b, v):
            return P(a, u) != P(b, v), f"{phrase(a, u).capitalize()} is not {phrase(b, v)}."
        if kind == "left" and T(a, u) + 1 == T(b, v):
            return P(a, u) + 1 == P(b, v), (f"{phrase(a, u).capitalize()} lives directly to "
                                            f"the left of {phrase(b, v)}.")
        if kind == "next" and abs(int(T(a, u)) - int(T(b, v))) == 1:
            return np.abs(P(a, u) - P(b, v)) == 1, (f"{phrase(a, u).capitalize()} lives next "
                                                    f"to {phrase(b, v)}.")
        if kind == "before" and T(a, u) < T(b, v):
            return P(a, u) < P(b, v), (f"{phrase(a, u).capitalize()} lives somewhere to the "
                                       f"left of {phrase(b, v)}.")
        if kind == "at":
            return P(a, u) == T(a, u), f"{phrase(a, u).capitalize()} lives in house {T(a, u) + 1}."
        if kind == "not_at":
            h = rng.choice([x for x in range(4) if x != T(a, u)])
            return P(a, u) != h, f"{phrase(a, u).capitalize()} does not live in house {h + 1}."
        return None

    clues, alive = [], np.ones(len(space), dtype=bool)
    while alive.sum() > 1:
        c = clue()
        if c is None or c[1] in (x[1] for x in clues):
            continue
        narrowed = alive & c[0]
        if narrowed.sum() < alive.sum():          # only clues that say something new
            clues.append(c)
            alive = narrowed
    # Drop every clue the others already imply: a minimal clue set is what
    # forces real deduction rather than reading the answer off one line.
    rng.shuffle(clues)
    k = 0
    while k < len(clues):
        rest = np.ones(len(space), dtype=bool)
        for j, c in enumerate(clues):
            if j != k:
                rest &= c[0]
        if rest.sum() == 1:
            clues.pop(k)
        else:
            k += 1
    rng.shuffle(clues)
    names_in_order = [ZEBRA_ATTRS["name"][int(np.where(truth[0] == h)[0][0])] for h in range(4)]
    clue_text = "\n".join(f"{n + 1}. {c[1]}" for n, c in enumerate(clues))
    query = ("Four houses stand in a row, numbered 1 to 4 from left to right. Each is home to "
             "one person, and the four people have different names (Alice, Bruno, Chen, Dara), "
             "drinks (coffee, tea, milk, juice), pets (cat, dog, parrot, fish) and jobs "
             f"(baker, doctor, pilot, teacher).\n\n{clue_text}\n\n"
             "Who lives in each house? Give the four names in house order, 1 to 4, "
             "separated by commas." + ANSWER_LINE.replace("<answer>", "<name>, <name>, <name>, <name>"))
    return _question("zebra", i, query, eval_method="answer_sequence", expected=names_in_order)


# --- shortest path ----------------------------------------------------------

def _dijkstra(adj, src, dst):
    dist, heap = {src: 0}, [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            return d
        if d > dist.get(u, 1e9):
            continue
        for v, w in adj[u]:
            if d + w < dist.get(v, 1e9):
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return None


def _greedy(adj, src, dst):
    """Always take the cheapest edge to an unvisited node -- the shortcut a
    model that doesn't search will take."""
    seen, u, total = {src}, src, 0
    while u != dst:
        options = [(w, v) for v, w in adj[u] if v not in seen]
        if not options:
            return None
        w, u = min(options)
        seen.add(u)
        total += w
    return total


def shortest_path(rng: random.Random, i: int) -> dict:
    nodes = [chr(ord("A") + k) for k in range(12)]
    while True:
        edges = {}
        for k in range(1, len(nodes)):                       # spanning tree: connected
            edges[tuple(sorted((nodes[k], nodes[rng.randrange(k)])))] = rng.randint(2, 30)
        while len(edges) < 24:
            a, b = rng.sample(nodes, 2)
            edges.setdefault(tuple(sorted((a, b))), rng.randint(2, 30))
        adj = {n: [] for n in nodes}
        for (a, b), w in edges.items():
            adj[a].append((b, w))
            adj[b].append((a, w))
        best = _dijkstra(adj, "A", "L")
        greedy = _greedy(adj, "A", "L")
        direct = edges.get(("A", "L"))
        if (greedy is None or greedy > best) and (direct is None or direct > best):
            break
    edge_list = "; ".join(f"{a}-{b}: {w}" for (a, b), w in sorted(edges.items()))
    query = ("An undirected road network connects towns A to L. Each road and its length:\n"
             f"{edge_list}\n\nWhat is the length of the shortest route from A to L?"
             + ANSWER_LINE.replace("<answer>", "<number>"))
    return _question("shortest_path", i, query, eval_method="answer_integer", expected=best)


# --- lattice paths ----------------------------------------------------------

def grid_paths(rng: random.Random, i: int) -> dict:
    size = 7
    while True:
        blocked = set()
        while len(blocked) < 6:
            p = (rng.randint(0, size), rng.randint(0, size))
            if p not in ((0, 0), (size, size)):
                blocked.add(p)
        ways = [[0] * (size + 1) for _ in range(size + 1)]
        for x in range(size + 1):
            for y in range(size + 1):
                if (x, y) in blocked:
                    continue
                if (x, y) == (0, 0):
                    ways[x][y] = 1
                    continue
                ways[x][y] = (ways[x - 1][y] if x else 0) + (ways[x][y - 1] if y else 0)
        if ways[size][size] >= 200:
            break
    pts = ", ".join(f"({x}, {y})" for x, y in sorted(blocked))
    query = (f"A walker starts at (0, 0) and must reach ({size}, {size}). Each step moves exactly "
             "one unit right (x + 1) or one unit up (y + 1). The walker may never stand on "
             f"any of these blocked points: {pts}.\n\nHow many different routes are there?"
             + ANSWER_LINE.replace("<answer>", "<number>"))
    return _question("grid_paths", i, query, eval_method="answer_integer", expected=ways[size][size])


# --- AIME -------------------------------------------------------------------

ROWS_URL = "https://datasets-server.huggingface.co/rows"
AIME_SOURCES = [("MathArena/aime_2025", "default", "train", "AIME 2025"),
                ("HuggingFaceH4/aime_2024", "default", "train", "AIME 2024")]


def aime(rng: random.Random) -> list[dict]:
    """Every AIME 2025 and 2024 problem, 2025 first: it postdates the
    training data of most models still in use, so a run that only gets
    through the first few is measuring reasoning rather than recall."""
    out = []
    for dataset, config, split, label in AIME_SOURCES:
        url = (f"{ROWS_URL}?dataset={urllib.parse.quote(dataset, safe='')}"
               f"&config={config}&split={split}&offset=0&length=100")
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                rows = [x["row"] for x in json.load(r)["rows"]]
        except Exception as e:                     # one source down is not fatal
            print(f"  skipped {label}: {e}")
            continue
        rng.shuffle(rows)
        for k, row in enumerate(rows):
            out.append({
                "id": f"x_aime{label[-4:]}_{row.get('problem_idx', k)}",
                "query": row["problem"],
                "answer_format": ANSWER_LINE.replace("<answer>", "<integer from 0 to 999>"),
                "difficulty": "expert", "task_type": "aime", "source": label,
                "eval_method": "answer_integer", "expected": int(str(row["answer"]).strip()),
            })
        print(f"  {label:15} {len(rows)}")
    return out


FAMILIES = {"countdown": countdown, "knights_knaves": knights_knaves, "zebra": zebra,
            "shortest_path": shortest_path, "grid_paths": grid_paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="aime",
                    help=f"comma-separated, from: aime, {', '.join(FAMILIES)}")
    ap.add_argument("--per-family", type=int, default=12, help="generated puzzles per family")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(EXPERT_QUERIES))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    wanted = args.families.split(",")
    questions = []
    space = _zebra_space() if "zebra" in wanted else None
    for family in wanted:
        if family == "aime":
            questions += aime(rng)
            continue
        for i in range(args.per_family):
            make = FAMILIES[family]
            questions.append(make(rng, i, space) if family == "zebra" else make(rng, i))
        print(f"  {family:15} {args.per_family}")

    with open(args.out, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"wrote {len(questions)} questions to {args.out}")


if __name__ == "__main__":
    main()
