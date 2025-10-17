/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef SILK_MACROS_H
#define SILK_MACROS_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "opus_types.h"
#include "opus_defines.h"
#include "arch.h"

/* This is an OPUS_INLINE header file for general platform. */

/* (a32 * (opus_int32)((opus_int16)(b32))) >> 16 output have to be 32bit int */
#if OPUS_FAST_INT64
#define silk_SMULWB(a32, b32)            ((opus_int32)(((a32) * (opus_int64)((opus_int16)(b32))) >> 16))
#else
#define silk_SMULWB(a32, b32)            ((((a32) >> 16) * (opus_int32)((opus_int16)(b32))) + ((((a32) & 0x0000FFFF) * (opus_int32)((opus_int16)(b32))) >> 16))
#endif

/* a32 + (b32 * (opus_int32)((opus_int16)(c32))) >> 16 output have to be 32bit int */
#if OPUS_FAST_INT64
#define silk_SMLAWB(a32, b32, c32)       ((opus_int32)((a32) + (((b32) * (opus_int64)((opus_int16)(c32))) >> 16)))
#else
#define silk_SMLAWB(a32, b32, c32)       ((a32) + ((((b32) >> 16) * (opus_int32)((opus_int16)(c32))) + ((((b32) & 0x0000FFFF) * (opus_int32)((opus_int16)(c32))) >> 16)))
#endif

/* (a32 * (b32 >> 16)) >> 16 */
#if OPUS_FAST_INT64
#define silk_SMULWT(a32, b32)            ((opus_int32)(((a32) * (opus_int64)((b32) >> 16)) >> 16))
#else
#define silk_SMULWT(a32, b32)            (((a32) >> 16) * ((b32) >> 16) + ((((a32) & 0x0000FFFF) * ((b32) >> 16)) >> 16))
#endif

/* a32 + (b32 * (c32 >> 16)) >> 16 */
#if OPUS_FAST_INT64
#define silk_SMLAWT(a32, b32, c32)       ((opus_int32)((a32) + (((b32) * ((opus_int64)(c32) >> 16)) >> 16)))
#else
#define silk_SMLAWT(a32, b32, c32)       ((a32) + (((b32) >> 16) * ((c32) >> 16)) + ((((b32) & 0x0000FFFF) * ((c32) >> 16)) >> 16))
#endif

/* (opus_int32)((opus_int16)(a3))) * (opus_int32)((opus_int16)(b32)) output have to be 32bit int */
#define silk_SMULBB(a32, b32)            ((opus_int32)((opus_int16)(a32)) * (opus_int32)((opus_int16)(b32)))

/* a32 + (opus_int32)((opus_int16)(b32)) * (opus_int32)((opus_int16)(c32)) output have to be 32bit int */
#define silk_SMLABB(a32, b32, c32)       ((a32) + ((opus_int32)((opus_int16)(b32))) * (opus_int32)((opus_int16)(c32)))

/* (opus_int32)((opus_int16)(a32)) * (b32 >> 16) */
#define silk_SMULBT(a32, b32)            ((opus_int32)((opus_int16)(a32)) * ((b32) >> 16))

/* a32 + (opus_int32)((opus_int16)(b32)) * (c32 >> 16) */
#define silk_SMLABT(a32, b32, c32)       ((a32) + ((opus_int32)((opus_int16)(b32))) * ((c32) >> 16))

/* a64 + (b32 * c32) */
#define silk_SMLAL(a64, b32, c32)        (silk_ADD64((a64), ((opus_int64)(b32) * (opus_int64)(c32))))

/* (a32 * b32) >> 16 */
#if OPUS_FAST_INT64
#define silk_SMULWW(a32, b32)            ((opus_int32)(((opus_int64)(a32) * (b32)) >> 16))
#else
#define silk_SMULWW(a32, b32)            silk_MLA(silk_SMULWB((a32), (b32)), (a32), silk_RSHIFT_ROUND((b32), 16))
#endif

/* a32 + ((b32 * c32) >> 16) */
#if OPUS_FAST_INT64
#define silk_SMLAWW(a32, b32, c32)       ((opus_int32)((a32) + (((opus_int64)(b32) * (c32)) >> 16)))
#else
#define silk_SMLAWW(a32, b32, c32)       silk_MLA(silk_SMLAWB((a32), (b32), (c32)), (b32), silk_RSHIFT_ROUND((c32), 16))
#endif

/* add/subtract with output saturated */
#define silk_ADD_SAT32(a, b)             ((((opus_uint32)(a) + (opus_uint32)(b)) & 0x80000000) == 0 ?                              \
                                        ((((a) & (b)) & 0x80000000) != 0 ? silk_int32_MIN : (a)+(b)) :   \
                                        ((((a) | (b)) & 0x80000000) == 0 ? silk_int32_MAX : (a)+(b)) )

#define silk_SUB_SAT32(a, b)             ((((opus_uint32)(a)-(opus_uint32)(b)) & 0x80000000) == 0 ?                                        \
                                        (( (a) & ((b)^0x80000000) & 0x80000000) ? silk_int32_MIN : (a)-(b)) :    \
                                        ((((a)^0x80000000) & (b)  & 0x80000000) ? silk_int32_MAX : (a)-(b)) )

#if defined(MIPSr1_ASM)
#include "mips/macros_mipsr1.h"
#endif

#include "ecintrin.h"
#ifndef OVERRIDE_silk_CLZ16
static OPUS_INLINE opus_int32 silk_CLZ16(opus_int16 in16)
{
    return 32 - EC_ILOG(in16<<16|0x8000);
}
#endif

#ifndef OVERRIDE_silk_CLZ32
static OPUS_INLINE opus_int32 silk_CLZ32(opus_int32 in32)
{
    return in32 ? 32 - EC_ILOG(in32) : 32;
}
#endif

/* Row based */
#define matrix_ptr(Matrix_base_adr, row, column, N) \
    (*((Matrix_base_adr) + ((row)*(N)+(column))))
#define matrix_adr(Matrix_base_adr, row, column, N) \
      ((Matrix_base_adr) + ((row)*(N)+(column)))

/* Column based */
#ifndef matrix_c_ptr
#   define matrix_c_ptr(Matrix_base_adr, row, column, M) \
    (*((Matrix_base_adr) + ((row)+(M)*(column))))
#endif

#ifdef OPUS_ARM_INLINE_ASM
#include "arm/macros_armv4.h"
#endif

#ifdef OPUS_ARM_INLINE_EDSP
#include "arm/macros_armv5e.h"
#endif

#ifdef OPUS_ARM_PRESUME_AARCH64_NEON_INTR
#include "arm/macros_arm64.h"
#endif

#endif /* SILK_MACROS_H */

