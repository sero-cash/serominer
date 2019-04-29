/*
This file is part of ethminer.

ethminer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ethminer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define DEV_INLINE __device__ __forceinline__

#ifdef __INTELLISENSE__
/* reduce vstudio warnings (__byteperm, blockIdx...) */
#include <device_functions.h>
#include <device_launch_parameters.h>
#define __launch_bounds__(max_tpb, min_blocks)
#define asm("a" : "=l"(result) : "l"(a))
#define __CUDA_ARCH__ 520  // highlight shuffle code by default.
#define __funnelshift_r(x, y)
#define __funnelshift_l(x, y)
#define __ldg(x)

uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t z);
uint32_t __shfl(uint32_t x, uint32_t y, uint32_t z);
uint32_t atomicExch(uint32_t* x, uint32_t y);
uint32_t atomicAdd(uint32_t* x, uint32_t y);
void __syncthreads(void);
void __threadfence(void);
void __threadfence_block(void);
#endif

#include <stdint.h>

DEV_INLINE uint64_t cuda_swab64(const uint64_t x)
{
    uint64_t result;
    uint2 t;
    asm("mov.b64 {%0,%1},%2; \n\t" : "=r"(t.x), "=r"(t.y) : "l"(x));
    t.x = __byte_perm(t.x, 0, 0x0123);
    t.y = __byte_perm(t.y, 0, 0x0123);
    asm("mov.b64 %0,{%1,%2}; \n\t" : "=l"(result) : "r"(t.y), "r"(t.x));
    return result;
}

DEV_INLINE uint64_t devectorize(uint2 x)
{
    uint64_t result;
    asm("mov.b64 %0,{%1,%2}; \n\t" : "=l"(result) : "r"(x.x), "r"(x.y));
    return result;
}

DEV_INLINE uint2 vectorize(const uint64_t x)
{
    uint2 result;
    asm("mov.b64 {%0,%1},%2; \n\t" : "=r"(result.x), "=r"(result.y) : "l"(x));
    return result;
}
DEV_INLINE void devectorize2(uint4 inn, uint2& x, uint2& y)
{
    x.x = inn.x;
    x.y = inn.y;
    y.x = inn.z;
    y.y = inn.w;
}

DEV_INLINE uint4 vectorize2(uint2 x, uint2 y)
{
    return make_uint4(x.x, x.y, y.x, y.y);
}

DEV_INLINE uint4 vectorize2(uint2 x)
{
    return make_uint4(x.x, x.y, x.x, x.y);
}
static DEV_INLINE uint2 operator^(uint2 a, uint32_t b)
{
    return make_uint2(a.x ^ b, a.y);
}
static DEV_INLINE uint2 operator^(uint2 a, uint2 b)
{
    return make_uint2(a.x ^ b.x, a.y ^ b.y);
}
static DEV_INLINE uint2 operator&(uint2 a, uint2 b)
{
    return make_uint2(a.x & b.x, a.y & b.y);
}
static DEV_INLINE uint2 operator|(uint2 a, uint2 b)
{
    return make_uint2(a.x | b.x, a.y | b.y);
}
static DEV_INLINE uint2 operator~(uint2 a)
{
    return make_uint2(~a.x, ~a.y);
}
static DEV_INLINE void operator^=(uint2& a, uint2 b)
{
    a = a ^ b;
}
static DEV_INLINE uint2 operator+(uint2 a, uint2 b)
{
    uint2 result;
    asm("{\n\t"
        "add.cc.u32 %0,%2,%4; \n\t"
        "addc.u32 %1,%3,%5;   \n\t"
        "}\n\t"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
    return result;
}

static DEV_INLINE uint2 operator+(uint2 a, uint32_t b)
{
    uint2 result;
    asm("{\n\t"
        "add.cc.u32 %0,%2,%4; \n\t"
        "addc.u32 %1,%3,%5;   \n\t"
        "}\n\t"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(a.y), "r"(b), "r"(0));
    return result;
}


static DEV_INLINE uint2 operator-(uint2 a, uint32_t b)
{
    uint2 result;
    asm("{\n\t"
        "sub.cc.u32 %0,%2,%4; \n\t"
        "subc.u32 %1,%3,%5;   \n\t"
        "}\n\t"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(a.y), "r"(b), "r"(0));
    return result;
}


static DEV_INLINE uint2 operator-(uint2 a, uint2 b)
{
    uint2 result;
    asm("{\n\t"
        "sub.cc.u32 %0,%2,%4; \n\t"
        "subc.u32 %1,%3,%5;   \n\t"
        "}\n\t"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
    return result;
}

static DEV_INLINE uint4 operator^(uint4 a, uint4 b)
{
    return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}
static DEV_INLINE uint4 operator&(uint4 a, uint4 b)
{
    return make_uint4(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w);
}
static DEV_INLINE uint4 operator|(uint4 a, uint4 b)
{
    return make_uint4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w);
}
static DEV_INLINE uint4 operator~(uint4 a)
{
    return make_uint4(~a.x, ~a.y, ~a.z, ~a.w);
}
static DEV_INLINE void operator^=(uint4& a, uint4 b)
{
    a = a ^ b;
}
static DEV_INLINE uint4 operator^(uint4 a, uint2 b)
{
    return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.x, a.w ^ b.y);
}
static DEV_INLINE void operator+=(uint2& a, uint2 b)
{
    a = a + b;
}

/**
 * basic multiplication between 64bit no carry outside that range (ie mul.lo.b64(a*b))
 * (what does uint64 "*" operator)
 */
static DEV_INLINE uint2 operator*(uint2 a, uint2 b)
{
    uint2 result;
    asm("{\n\t"
        "mul.lo.u32        %0,%2,%4;  \n\t"
        "mul.hi.u32        %1,%2,%4;  \n\t"
        "mad.lo.cc.u32    %1,%3,%4,%1; \n\t"
        "madc.lo.u32      %1,%3,%5,%1; \n\t"
        "}\n\t"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
    return result;
}

DEV_INLINE uint32_t ROL8(const uint32_t x)
{
    return __byte_perm(x, x, 0x2103);
}

DEV_INLINE uint2 ROL8(const uint2 a)
{
    return make_uint2(__byte_perm(a.y, a.x, 0x6543), __byte_perm(a.y, a.x, 0x2107));
}

DEV_INLINE uint2 ROR8(const uint2 a)
{
    return make_uint2(__byte_perm(a.y, a.x, 0x0765), __byte_perm(a.y, a.x, 0x4321));
}

#if __CUDA_ARCH__ >= 350
__inline__ __device__ uint2 ROL2(const uint2 a, const int offset)
{
    uint2 result;
    if (offset >= 32)
    {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
    }
    else
    {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
    }
    return result;
}
#else
__inline__ __device__ uint2 ROL2(const uint2 v, const int n)
{
    uint2 result;
    if (n <= 32)
    {
        result.y = ((v.y << (n)) | (v.x >> (32 - n)));
        result.x = ((v.x << (n)) | (v.y >> (32 - n)));
    }
    else
    {
        result.y = ((v.x << (n - 32)) | (v.y >> (64 - n)));
        result.x = ((v.y << (n - 32)) | (v.x >> (64 - n)));
    }
    return result;
}
#endif

DEV_INLINE uint32_t bfe(uint32_t x, uint32_t bit, uint32_t numBits)
{
    uint32_t ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
    return ret;
}
