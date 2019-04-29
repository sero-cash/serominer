#define OPENCL_PLATFORM_UNKNOWN 0
#define OPENCL_PLATFORM_NVIDIA  1
#define OPENCL_PLATFORM_AMD     2
#define OPENCL_PLATFORM_CLOVER  3
#define OPENCL_PLATFORM_APPLE   4

#ifndef MAX_SEARCH_RESULTS
#define MAX_SEARCH_RESULTS 4U
#endif

#ifndef PLATFORM
#define PLATFORM OPENCL_PLATFORM_AMD
#endif

#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif

#define HASHES_PER_GROUP (GROUP_SIZE / PROGPOW_LANES)

typedef struct
{
    uint32_t uint32s[32 / sizeof(uint32_t)];
} hash32_t;

// Implementation based on:
// https://github.com/mjosaarinen/tiny_sha3/blob/master/sha3.c

__constant const uint32_t keccakf_rndc[24] = {
    0x00000001, 0x00008082, 0x0000808a, 0x80008000, 0x0000808b, 0x80000001,
    0x80008081, 0x00008009, 0x0000008a, 0x00000088, 0x80008009, 0x8000000a,
    0x8000808b, 0x0000008b, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800a, 0x8000000a, 0x80008081, 0x00008080, 0x80000001, 0x80008008
};

// Implementation of the Keccakf transformation with a width of 800
void keccak_f800_round(uint32_t st[25], const int r)
{

    const uint32_t keccakf_rotc[24] = {
        1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
        27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
    };
    const uint32_t keccakf_piln[24] = {
        10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
        15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
    };

    uint32_t t, bc[5];
    // Theta
    for (int i = 0; i < 5; i++)
        bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

    for (int i = 0; i < 5; i++) {
        t = bc[(i + 4) % 5] ^ ROTL32(bc[(i + 1) % 5], 1u);
        for (uint32_t j = 0; j < 25; j += 5)
            st[j + i] ^= t;
    }

    // Rho Pi
    t = st[1];
    for (int i = 0; i < 24; i++) {
        uint32_t j = keccakf_piln[i];
        bc[0] = st[j];
        st[j] = ROTL32(t, keccakf_rotc[i]);
        t = bc[0];
    }

    //  Chi
    for (uint32_t j = 0; j < 25; j += 5) {
        for (int i = 0; i < 5; i++)
            bc[i] = st[j + i];
        for (int i = 0; i < 5; i++)
            st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }

    //  Iota
    st[0] ^= keccakf_rndc[r];
}

// Keccak - implemented as a variant of SHAKE
// The width is 800, with a bitrate of 576, a capacity of 224, and no padding
// Only need 64 bits of output for mining
uint64_t keccak_f800(__constant hash32_t const* g_header, uint64_t seed, hash32_t digest)
{
    uint32_t st[25] = {
        g_header->uint32s[0],
        g_header->uint32s[1],
        g_header->uint32s[2],
        g_header->uint32s[3],
        g_header->uint32s[4],
        g_header->uint32s[5],
        g_header->uint32s[6],
        g_header->uint32s[7],
        seed,
        seed >> 32,
        digest.uint32s[0],
        digest.uint32s[1],
        digest.uint32s[2],
        digest.uint32s[3],
        digest.uint32s[4],
        digest.uint32s[5],
        digest.uint32s[6],
        digest.uint32s[7],
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    };

    for (int r = 0; r < 22; r++) {
        keccak_f800_round(st, r);
    }

    // TODO elaborate this
    // last round can be simplified due to partial output
    // keccak_f800_round(st, 21);

    // Byte swap so byte 0 of hash is MSB of result
    uint64_t res = (uint64_t)st[1] << 32 | st[0];
    return as_ulong(as_uchar8(res).s76543210);
}

#define fnv1a(h, d) (h = (h ^ d) * 0x1000193)

typedef struct {
    uint32_t z, w, jsr, jcong;
} kiss99_t;

// KISS99 is simple, fast, and passes the TestU01 suite
// https://en.wikipedia.org/wiki/KISS_(algorithm)
// http://www.cse.yorku.ca/~oz/marsaglia-rng.html
uint32_t kiss99(kiss99_t *st)
{
    st->z = 36969 * (st->z & 65535) + (st->z >> 16);
    st->w = 18000 * (st->w & 65535) + (st->w >> 16);
    uint32_t MWC = ((st->z << 16) + st->w);
    st->jsr ^= (st->jsr << 17);
    st->jsr ^= (st->jsr >> 13);
    st->jsr ^= (st->jsr << 5);
    st->jcong = 69069 * st->jcong + 1234567;
    return ((MWC^st->jcong) + st->jsr);
}

void fill_mix(uint64_t seed, uint32_t lane_id, uint32_t mix[PROGPOW_REGS])
{
    // Use FNV to expand the per-warp seed to per-lane
    // Use KISS to expand the per-lane seed to fill mix
    uint32_t fnv_hash = 0x811c9dc5;
    kiss99_t st;
    st.z = fnv1a(fnv_hash, seed);
    st.w = fnv1a(fnv_hash, seed >> 32);
    st.jsr = fnv1a(fnv_hash, lane_id);
    st.jcong = fnv1a(fnv_hash, lane_id);
    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++)
        mix[i] = kiss99(&st);
}

typedef struct {
    uint32_t uint32s[PROGPOW_LANES];
    uint64_t uint64s[PROGPOW_LANES / 2];
} shuffle_t;

// NOTE: This struct must match the one defined in CLMiner.cpp
typedef struct
{
    uint count;
    uint abort;
    uint rounds;
    struct
    {
        // One word for gid and 8 for mix hash
        ulong nonce;
        uint mix[8];
    } result[MAX_SEARCH_RESULTS];
} search_results;

#if PLATFORM != OPENCL_PLATFORM_NVIDIA // use maxrregs on nv
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
#endif
__kernel void progpow_search(
    __global volatile search_results* restrict g_output,
    __constant hash32_t const* g_header,
    __global ulong8 const* _g_dag,
    ulong start_nonce,
    __global ulong const* restrict target,
    uint hack_false
)
{

    if (g_output->abort)
        return;
    if (get_local_id(0) == 0)
        atomic_inc(&g_output->rounds);

    __global dag_t const* g_dag = (__global dag_t const*)_g_dag;

    __local shuffle_t share[HASHES_PER_GROUP];
    __local uint32_t c_dag[PROGPOW_CACHE_WORDS];

    uint32_t const lid = get_local_id(0);
    uint32_t const gid = get_global_id(0);
    uint64_t const nonce = start_nonce + gid;

    const uint32_t lane_id = lid & (PROGPOW_LANES - 1);
    const uint32_t group_id = lid / PROGPOW_LANES;

    // Load the first portion of the DAG into the cache
    for (uint32_t word = lid*PROGPOW_DAG_LOADS; word < PROGPOW_CACHE_WORDS; word += GROUP_SIZE*PROGPOW_DAG_LOADS)
    {
        dag_t load = g_dag[word/PROGPOW_DAG_LOADS];
        for (int i = 0; i<PROGPOW_DAG_LOADS; i++)
            c_dag[word + i] = load.s[i];
    }

    hash32_t digest;
    for (int i = 0; i < 8; i++)
        digest.uint32s[i] = 0;
    // keccak(header..nonce)
    uint64_t seed = keccak_f800(g_header, nonce, digest);

    barrier(CLK_LOCAL_MEM_FENCE);
    uint32_t mix[PROGPOW_REGS];

    #pragma unroll 1
    for (uint32_t h = 0; h < PROGPOW_LANES; h++)
    {

        // share the hash's seed across all lanes
        if (lane_id == h)
            share[group_id].uint64s[0] = seed;
        barrier(CLK_LOCAL_MEM_FENCE);
        uint64_t hash_seed = share[group_id].uint64s[0];

        // initialize mix for all lanes
        fill_mix(hash_seed, lane_id, mix);

        // For whatver reason prevent unrolling this loop causes
        // bogus periods on AMD OpenCL. Use any unroll factor greater than 1
        #pragma unroll 2
        for (uint32_t l = 0; l < PROGPOW_CNT_DAG; l++)
            progPowLoop(l, mix, g_dag, c_dag, share[0].uint64s, hack_false, lane_id, group_id);

        // Reduce mix data to a per-lane 32-bit digest
        uint32_t mix_hash = 0x811c9dc5;
        #pragma unroll
        for (int i = 0; i < PROGPOW_REGS; i++)
            fnv1a(mix_hash, mix[i]);

        // Reduce all lanes to a single 256-bit digest
        hash32_t digest_temp;
        for (int i = 0; i < 8; i++)
            digest_temp.uint32s[i] = 0x811c9dc5;
        share[group_id].uint32s[lane_id] = mix_hash;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int i = 0; i < PROGPOW_LANES; i++)
            fnv1a(digest_temp.uint32s[i & 7], share[group_id].uint32s[i]);
        if (h == lane_id)
            digest = digest_temp;
    }

    // keccak(header .. keccak(header..nonce) .. digest);
    if (keccak_f800(g_header, seed, digest) > *target)
        return;

    uint slot = atomic_inc(&g_output->count);
    if (slot >= MAX_SEARCH_RESULTS) 
        return;

    atomic_inc(&g_output->abort);
    g_output->result[slot].nonce = nonce;

    #pragma unroll
    for (int i = 0; i < 8; i++)
        g_output->result[slot].mix[i] = digest.uint32s[i];

}
