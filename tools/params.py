#!/usr/bin/env python3

# Useful commands for interpreting JSON output:
# `jq 'group_by(.log_p) | map(min_by(.query_sz))'`
# `jq 'group_by(.log_p) | map(min_by([.query_sz, -.noise_margin, .ell_ks]))'`

import sys

if not sys.version_info >= (3, 12):
    print('Error: Python 3.12 or newer is required', file=sys.stderr)
    exit(1)

import argparse
from itertools import batched, chain
import json
from math import ceil, floor, log, log2, sqrt
from tabulate import tabulate

DEFAULT_DATABASE_SIZE = 30

# 1+log2(sqrt(2*log(2/2**-40)))
DEFAULT_MIN_NOISE_MARGIN = 4

# Other terminology:
# pf: Pack Factor, the degree of unpacking using one automorphism key
# pd: Pack Depth, the number of automorphism keys
# ptot: pf**pd, the total number of packed ciphertexts per query ciphertext

class NoFiltersAction(argparse.Action):
    def __init__(self, option_strings, dest, help=None):
        self._option_strings = option_strings
        super().__init__(option_strings=option_strings, dest=dest, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self._option_strings:
            setattr(namespace, 'max_query_size', None)
            setattr(namespace, 'min_noise_margin', None)
            setattr(namespace, 'min_score', None)

    def format_usage(self):
        return ' | '.join(self._option_strings)

parser = argparse.ArgumentParser()
parser.add_argument('--json', action='store_true')
parser.add_argument('--best-json', action='store_true')
parser.add_argument('-n', '--polynomial-degree', type=int, choices=[1024, 2048, 4096])
parser.add_argument('-d', '--log-database-size', type=int, default=DEFAULT_DATABASE_SIZE)
parser.add_argument('--extended-sweep', action='store_true')
parser.add_argument('--log-p', type=int, help='use only the specified value for log_p')
parser.add_argument('--log-q', type=int, help='use only the specified value for log_q')
parser.add_argument('--log-slice-count', type=int, help='use only the specified value for log_slice_count')
parser.add_argument('--sigma', type=float, default=6.4)
parser.add_argument('--total-comm', action='store_true')
parser.add_argument('--sort', type=str, choices=['score', 'cost', 'comm'], default='score')
filter_group = parser.add_argument_group('Filtering options')
query_size_group = filter_group.add_mutually_exclusive_group()
query_size_group.add_argument('--max-query-size', type=int, default=1000)
query_size_group.add_argument('--no-max-query-size', action='store_const', dest='max_query_size')
noise_margin_group = filter_group.add_mutually_exclusive_group()
noise_margin_group.add_argument('--min-noise-margin', type=float, default=DEFAULT_MIN_NOISE_MARGIN)
noise_margin_group.add_argument('--no-min-noise-margin', action='store_const', dest='min_noise_margin')
score_group = filter_group.add_mutually_exclusive_group()
score_group.add_argument('--min-score', type=int, default=30)
score_group.add_argument('--no-min-score', action='store_const', dest='min_score')
filter_group.add_argument('--no-filters', action=NoFiltersAction, help='disable filtering by query size, noise margin, or score')
args = parser.parse_args()

# Old (log_q, log_slice_count) pairings: (31, 6), (63, 5), (127, 2)
if args.log_q is None:
    if args.polynomial_degree is None:
        rings = [(30, 1024), (56, 2048)]
    elif args.polynomial_degree == 1024:
        rings = [(30, 1024)]
    elif args.polynomial_degree == 2048:
        rings = [(56, 2048)]
    elif args.polynomial_degree == 4096:
        rings = [(108, 4096)]
    else:
        raise AssertionError(f'n = {args.polynomial_degree} should have been rejected by argparse')
else:
    if args.polynomial_degree is not None:
        rings = [(args.log_q, args.polynomial_degree)]
    elif args.log_q <= 32:
        rings = [(args.log_q, 1024)]
    elif args.log_q <= 64:
        rings = [(args.log_q, 2048)]
    else:
        rings = [(args.log_q, 4096)]

# in MiB/s
BFV_THROUGHPUT = 25_000
# in MiB/s, for ell_gsw = 1.
GSW_THROUGHPUT = 185
# in automorphisms/s, for ell_gsw = 1.
UNPACK_THROUGHPUT = 90_000

base_noise = log2(args.sigma)

def ext_prod_noise(n, log_mu_gsw, t_gsw, ell_gsw, noise_lwe, noise_gsw):
    term_a = noise_gsw + t_gsw + log2(n) / 2 + log2(ell_gsw) / 2 - 1
    term_b = noise_lwe + log_mu_gsw
    return log2(sqrt(4 ** term_a + 4 ** term_b))

# Table 1 from WhisPIR; de Castro, Lewi, and Suh, 2024.
unpack_mode1_iters = [
    (64, 192),
    (128, 2496),
    (256, 7168),
    (512, 20736),
    (1024, 113664),
    (2048, 386048),
]

# Key is (pf, pd, ksk_count)
# Value is (unpack_op_depth, unpack_op_count)
# unpack_op_depth is the depth of automorphisms in unpacking.
# In the case that one \tau_i Frobenius automorphism is iterated to implement
# another, unpack_op_depth may be larger than the number of homomorphic
# additions (`x + \tau_i(x)). The number of additions is always equal to `(pf -
# 1) * pd`.
packings = {
    (2, 1, 1): (1, 1),
    (2, 2, 1): (3, 4),
    (2, 2, 2): (2, 3),
    (2, 3, 1): (7, 12),
    (2, 3, 3): (3, 7),
    (2, 4, 1): (15, 32),
    (2, 4, 2): (6, 20),
    (2, 4, 4): (4, 15),
    (2, 5, 1): (31, 80),
    (2, 5, 5): (5, 31),
    (2, 6, 1): (63, 192),
    (2, 6, 2): (14, 108),
    (2, 6, 6): (6, 63),
    (2, 7, 1): (127, 448),
    (2, 7, 7): (7, 127),
    (2, 8, 1): (255, 1024),
    (2, 8, 2): (30, 544),
    (2, 8, 8): (8, 255),
}

# Ciphertext size, in bytes
def ct_size(log_q, n):
    return n * log_q / 8

# GSW mode 0: no synthesis. submit 2 * ell_gsw * gsw_bits ciphertexts
# GSW mode 1: submit ell_gsw * gsw_bits packed LWE ciphertexts and RGSW(-s) ciphertext.
#   Total size: ell_gsw * gsw_bits / pf ** pd + 2 * ell_gsw
def calculate(n, log_q, log_p, t_gsw, t_ks, slices, index_bits, packing, gsw_bits, gsw_mode):
    (pf, pd, ksk_count) = packing

    # t_gsw = decomposition factor for GSW ciphertexts.
    if t_gsw is not None:
        Beta = 2**t_gsw
        ell_gsw = ceil(log_q / t_gsw)
    else:
        assert(gsw_bits == 0)
        Beta = None
        ell_gsw = None
    # t_ks = decomposition factor for automorphism keys.
    if pd is not None:
        Beta_b = 2**t_ks
        ell_ks = ceil(log_q / t_ks)
        ptot = pf ** pd
    else:
        assert(t_ks is None)
        Beta_b = None
        ell_ks = None
        ptot = 1

    ct_size_kb = ct_size(log_q, n) / 1024

    assert(ptot <= n / 2)

    # Another possible GSW mode is to send 1 * ell_gsw ciphertexts for each GSW
    # bit along with an RGSW(-s) ciphertext, and derive the other half of the
    # ciphertexts. The downside of this is additional noise in the derived
    # ciphertexts.
    if gsw_bits == 0:
        packed_ctexts_for_gsw = 0
        unpacked_ctexts = 0
    elif gsw_mode == 0:
        packed_ctexts_for_gsw = 0
        unpacked_ctexts = 2 * ell_gsw * gsw_bits
    elif gsw_mode == 1:
        packed_ctexts_for_gsw = ell_gsw * gsw_bits
        unpacked_ctexts = 2 * ell_gsw
    else:
        raise Exception(f"Unknown GSW mode {gsw_mode}")
    if ell_ks is not None:
        ksk_ctexts = ell_ks * ksk_count
    else:
        ksk_ctexts = 0

    i0_bits = index_bits - gsw_bits
    packed_ctexts = ceil((2.0**i0_bits + packed_ctexts_for_gsw) / ptot)

    # This does not reflect 50% savings on modulus-switched GSW ciphertexts
    # 1: Number of query LWE ciphertexts (each contains ptot packed ciphertexts, which
    #    may be used for GSW synthesis, or directly as query ciphertexts)
    # 2. Key switch keys for unpacking LWE ciphertexts
    # 3. GSW ciphertexts
    query_size = ct_size_kb * (ksk_ctexts + packed_ctexts + unpacked_ctexts)
    if log_q <= 32 and log_p <= 4:
        mod_switch_log_q = 14
    elif log_q <= 32 and log_p <= 8:
        mod_switch_log_q = 20
    elif log_q <= 32:
        mod_switch_log_q = log_p + 12
    elif log_q <= 64 and log_p <= 8:
        mod_switch_log_q = 21
    elif log_q <= 64:
        mod_switch_log_q = log_p + 13
    else:
        raise ValueError(f'modulus switching factor is unknown for log_q = {log_q} / log_p = {log_p}')
    modulus_switch_factor = mod_switch_log_q / log_q
    response_size = modulus_switch_factor * 2 * ct_size_kb * slices

    if args.total_comm:
        comm = query_size + response_size
    else:
        comm = 2 * query_size
        if response_size > 1.25 * query_size:
            return

    # The multiplicative term should really by (pf - 1) * pd, but the
    # noise growth in early iterations is higher because the key switch input
    # ciphertext has noise comparable to the key-switch-induced noise. So, we
    # fudge the term to roughly compensate for that.
    if pd is not None:
        try:
            (unpack_op_depth, unpack_op_count) = packings[packing]
        except KeyError:
            raise ValueError(f'packing {packing} is not characterized')

        if ksk_count == pd:
            unpack_ks_depth = (ptot ** 2 - 1) / 3
        elif ksk_count == 1:
            unpack_ks_depth = (ptot ** 2 - ptot) / 2
        elif ksk_count == 2:
            hpd = pd // 2
            unpack_ks_depth = (pf**hpd + 1) * ((pf ** hpd) ** 2 - (pf ** hpd)) / 2
            #print(f'pf = {pf} pd = {pd} ks_depth = {unpack_ks_depth}')
        else:
            raise ValueError(f'ksk_count {ksk_count} is not supported')
        unpack_noise = log2(sqrt(ptot * 4**base_noise + unpack_ks_depth * 4**(base_noise + t_ks + log2(n) / 2 + log2(ell_ks) / 2 - 1)))
    else:
        unpack_op_depth = None
        unpack_op_count = None
        unpack_noise = base_noise

    if gsw_bits == 0:
        gsw_noise = None
    elif gsw_mode == 0:
        gsw_noise = base_noise
    elif gsw_mode == 1:
        gsw_noise = ext_prod_noise(n, base_noise + log2(n) / 2, t_gsw, ell_gsw, unpack_noise, base_noise)
    else:
        raise Exception(f"Unknown GSW mode {gsw_mode}")

    p1_noise = unpack_noise + i0_bits / 2 + log2(n) / 2 + log_p

    answer_noise = p1_noise
    for _ in range(0, gsw_bits):
        answer_noise = ext_prod_noise(n, 0, t_gsw, ell_gsw, answer_noise, gsw_noise)

    # Note: GSW-ciphertext noise term (~ t) is assumed not to dominate and ignored.
    #  -> Is this a valid assumption even with large gsw_bits? Maybe
    #     (gsw_bits / 2 + t), so 20.5 for the likely "small" parameters?
    # Why ix_bits / 2? sqrt because independent?
    #   i0/2 for unpacking, i0/2 for DB reduction, ix/2 for DB reduction?
    # Why no term for noise distribution (binary noise?)
    noise_margin = log_q - log_p - answer_noise

    # Warning: This does not include post-BFV inverse NTT
    if log_q <= 32:
        unpack_throughput = UNPACK_THROUGHPUT
    elif log_q <= 64:
        unpack_throughput = UNPACK_THROUGHPUT / 2
    else:
        unpack_throughput = UNPACK_THROUGHPUT / 4
    compute_bfv = 2.0**(args.log_database_size - 23) * log_q / log_p / BFV_THROUGHPUT
    if gsw_bits != 0:
        compute_gsw = 2.0**(args.log_database_size - 23) * log_q / log_p / 2**i0_bits * ell_gsw / GSW_THROUGHPUT
    else:
        compute_gsw = 0
    est_compute = compute_bfv + compute_gsw
    if pd is None:
        compute_unpack = None
    else:
        compute_unpack = packed_ctexts * unpack_op_count * ell_ks / unpack_throughput
        est_compute += compute_unpack

    cost = (13.9 * est_compute + 0.1 * comm) * 1e9 / 2.0**args.log_database_size
    score = 2.0**(args.log_database_size - 28) * 1_000_000 / est_compute / comm**2

    if args.min_noise_margin is not None and noise_margin < args.min_noise_margin:
        return
    if args.max_query_size is not None and query_size > args.max_query_size:
        return
    if args.min_score is not None and score < args.min_score:
        return

    noise_detail = {
        "unpack_noise": unpack_noise,
        "p1_noise": p1_noise,
        "gsw_noise": gsw_noise,
        "answer_noise": answer_noise,
    }

    unpack_detail = {
        "op_depth": unpack_op_depth,
        "op_count": unpack_op_count,
    }

    query_detail = {
        "packed_ctexts": packed_ctexts,
        "unpacked_ctexts": unpacked_ctexts,
        "ksk_ctexts": ksk_ctexts,
    }

    compute_detail = {
        "bfv": compute_bfv,
        "gsw": compute_gsw,
        "unpack": compute_unpack,
    }

    if gsw_mode == 1:
        latex_gsw_syn = '\\y'
    else:
        latex_gsw_syn = '  '
    latex = " {} & {:2} & {:2} & {:2} & {:2} & {:2}   & {:2} & {:2} & {:3} & {:2} & {:3.0f} & {:3.0f} \\ % {:.2f}, {:.0f}, {:.0f}".format(
        log_q,
        log_p,
        slices,
        ell_gsw if ell_gsw is not None else '-',
        ell_ks if ell_ks is not None else '-',
        index_bits,
        gsw_bits,
        latex_gsw_syn,
        ptot,
        mod_switch_log_q,
        query_size,
        response_size,
        est_compute,
        score,
        cost,
    )

    if gsw_mode == 1:
        rust_gsw_syn = f'Some({gsw_bits})'
    else:
        rust_gsw_syn = 'None'
    rust_db_size = "1 << {}".format(args.log_database_size - 3)
    rust = f"pir::<Lwe{n}Q{log_q}P{log_p}, {ell_gsw}, {ell_ks}, Lwe{n}Q{mod_switch_log_q}P{log_p}>({rust_db_size}, {slices}, {ptot}, {packed_ctexts}, {rust_gsw_syn});"

    return {
        "log_q": log_q,
        "log_p": log_p,
        "t_gsw": t_gsw,
        "t_ks": t_ks,
        "l_gsw": ell_gsw,
        "l_ks": ell_ks,
        "slices": slices,
        "i_bits": index_bits,
        "pf": pf,
        "pd": pd,
        "n_ksk": ksk_count,
        "i0_bits": i0_bits,
        "g_bits": gsw_bits,
        "g_mode": gsw_mode,
        "nse": answer_noise,
        "n_mgn": noise_margin,
        "query_sz": query_size,
        "resp_sz": response_size,
        "tot_sz": query_size + response_size,
        "noise_detail": noise_detail,
        "query_detail": query_detail,
        "compute_detail": compute_detail,
        "comp": est_compute,
        "score": score,
        "cost": cost,
        "latex": latex,
        "rust": rust,
    }

results = []
#print(json.dumps(calculate(2048, 56, 8, 10, 7, 4, 13, ( 8, 2, 1), 8, 1), indent=2)) # q56x
#print(json.dumps(calculate(2048, 56, 4, 12, 7, 4, 13, ( 8, 2, 1), 6, 1), indent=2)) # q56_pack256_c
#print(json.dumps(calculate(2048, 56, 4, 12, 7, 4, 13, ( 2, 4, 1), 4, 0), indent=2)) # experiment
#print(json.dumps(calculate(2048, 56, 4, 14, 7, 4, 13, (16, 2, 1), 4, 0), indent=2))
#print(json.dumps(calculate(2048, 56, 8,  8, 8, 4, 12, (16, 2, 1), 4, 1), indent=2))
#print(json.dumps(calculate(1024, 30, 6, 10, None, 6, 12, (1, None, None), 7, 0), indent=2))
#print(json.dumps(calculate(1024, 30, 1, 15, 3, 6, 14, ( 8, 2, 1), 6, 0), indent=2)) # q30_p1

#sys.exit(0)

def t_values(log_q, ks):
    if log_q <= 32 or args.extended_sweep:
        this_t = 1
    elif log_q <= 64:
        this_t = 1 if ks else 3
    else:
        this_t = 8 if ks else 16
    while this_t <= ceil(log_q / 2):
        yield this_t
        last_ell = ceil(log_q / this_t)
        this_t += 1
        while ceil(log_q / this_t) == last_ell:
            this_t += 1

for (log_q, n) in rings:
    # If restricted to power of two, log_p can be at most one quarter of log_q, because
    # it shows up 2x in the noise margin.
    if args.log_p is None:
        log_p_values = filter(lambda val: val <= (log_q + 1) / 4, [1, 2, 4, 6, 8, 12, 16, 24, 32])
    elif args.log_p > (log_q + 1) / 4:
        raise ValueError(f'log_p of {args.log_p} is too large for log_q = {log_q}')
    else:
        log_p_values = [args.log_p]

    for log_p in log_p_values:
        log_log_p = int(floor(log2(log_p)))
        log_log_q = ceil(log2(log_q))
        # index_bits calculation is only valid _without_ modulus switching for the
        # response. With modulus switching for the response, need to rethink dependency
        # on p and q.
        log_ct_size_bits = ceil(log2(ct_size(log_q, n) * 8))
        # This can be used to search log_slice_count, but that doesn't seem to
        # produce interesting configs. Varying gsw_bits has similar effect on
        # the compute tradeoff.
        if args.log_slice_count is not None:
            log_slice_count_values = [args.log_slice_count]
        else:
            log_slice_count_values = range(0, 17 - 2 * log_log_q)
        for log_slice_count in log_slice_count_values:
            index_bits = args.log_database_size - log_slice_count - log_ct_size_bits + log_log_q - log_log_p
            for gsw_bits in range(0, index_bits + 1):
                if gsw_bits != 0:
                    t_gsw_values = t_values(log_q, False)
                else:
                    t_gsw_values = [None]
                for t_gsw in t_gsw_values:
                    for packing in chain([(1, None, None)], packings.keys()):
                        if packing == (1, None, None):
                            t_ks_values = [None]
                        else:
                            t_ks_values = t_values(log_q, True)
                        for t_ks in t_ks_values:
                            result = calculate(n, log_q, log_p, t_gsw, t_ks, 2**log_slice_count, index_bits, packing, gsw_bits, 0)
                            if result is not None:
                                results.append(result)
                            if gsw_bits != 0:
                                result = calculate(n, log_q, log_p, t_gsw, t_ks, 2**log_slice_count, index_bits, packing, gsw_bits, 1)
                                if result is not None:
                                    results.append(result)

if args.sort == 'score':
    results.sort(key=lambda r: r['score'], reverse=True)
elif args.sort == 'comm':
    results.sort(key=lambda r: r['query_sz'] + r['resp_sz'])
else:
    results.sort(key=lambda r: r['cost'])

if args.best_json:
    best = []
    by_cost = results.copy()
    by_cost.sort(key=lambda r: r['cost'])
    by_query_sz = results.copy()
    by_query_sz.sort(key=lambda r: r['query_sz'])
    by_total_comm = results.copy()
    by_total_comm.sort(key=lambda r: r['query_sz'] + r['resp_sz'])
    by_compute = results.copy()
    by_compute.sort(key=lambda r: r['comp'])

    for (log_q, _) in rings:
        if args.total_comm:
            best.append(next(filter(lambda r: r['log_q'] == log_q, by_cost)))
            best.append(next(filter(lambda r: r['log_q'] == log_q, by_total_comm)))
            best.append(next(filter(lambda r: r['log_q'] == log_q, by_compute)))
        else:
            best.append(next(filter(lambda r: r['log_q'] == log_q, by_cost)))
            best.append(next(filter(lambda r: r['log_q'] == log_q, by_query_sz)))

    results = best

if args.json or args.best_json:
    json.dump(results, sys.stdout)
else:
    def massage_results(r):
        del r['noise_detail']
        del r['query_detail']
        del r['compute_detail']
        del r['latex']
        del r['rust']
        r['nse'] = round(r['nse'], 1)
        r['n_mgn'] = round(r['n_mgn'], 1)
        r['comp'] = round(r['comp'], 2)
        r['score'] = round(r['score'])
        r['cost'] = round(r['cost'])
        for k, v in r.items():
            if v is None:
                r[k] = '-'
        return r
    first = True
    for results_batch in batched(results, 50):
        if not first:
            print()
        print(tabulate(map(massage_results, results_batch), headers="keys", stralign="right"))
        first = False
