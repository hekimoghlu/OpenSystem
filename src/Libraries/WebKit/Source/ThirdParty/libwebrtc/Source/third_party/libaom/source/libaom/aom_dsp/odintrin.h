/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
/* clang-format off */

#ifndef AOM_AOM_DSP_ODINTRIN_H_
#define AOM_AOM_DSP_ODINTRIN_H_

#include <stdlib.h>
#include <string.h>

#include "aom/aom_integer.h"
#include "aom_dsp/aom_dsp_common.h"
#include "aom_ports/bitops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int od_coeff;

#define OD_DIVU_DMAX (1024)

extern uint32_t OD_DIVU_SMALL_CONSTS[OD_DIVU_DMAX][2];

/*Translate unsigned division by small divisors into multiplications.*/
#define OD_DIVU_SMALL(_x, _d)                                     \
  ((uint32_t)((OD_DIVU_SMALL_CONSTS[(_d)-1][0] * (uint64_t)(_x) + \
               OD_DIVU_SMALL_CONSTS[(_d)-1][1]) >>                \
              32) >>                                              \
   (OD_ILOG_NZ(_d) - 1))

#define OD_DIVU(_x, _d) \
  (((_d) < OD_DIVU_DMAX) ? (OD_DIVU_SMALL((_x), (_d))) : ((_x) / (_d)))

#define OD_MINI AOMMIN
#define OD_MAXI AOMMAX
#define OD_CLAMPI(min, val, max) (OD_MAXI(min, OD_MINI(val, max)))

/*Integer logarithm (base 2) of a nonzero unsigned 32-bit integer.
  OD_ILOG_NZ(x) = (int)floor(log2(x)) + 1.*/
#define OD_ILOG_NZ(x) (1 + get_msb(x))

/*Enable special features for gcc and compatible compilers.*/
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__)
#define OD_GNUC_PREREQ(maj, min, pat)                                \
  ((__GNUC__ << 16) + (__GNUC_MINOR__ << 8) + __GNUC_PATCHLEVEL__ >= \
   ((maj) << 16) + ((min) << 8) + pat)  // NOLINT
#else
#define OD_GNUC_PREREQ(maj, min, pat) (0)
#endif

#if OD_GNUC_PREREQ(3, 4, 0)
#define OD_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))
#else
#define OD_WARN_UNUSED_RESULT
#endif

#if OD_GNUC_PREREQ(3, 4, 0)
#define OD_ARG_NONNULL(x) __attribute__((__nonnull__(x)))
#else
#define OD_ARG_NONNULL(x)
#endif

/*All of these macros should expect floats as arguments.*/
# define OD_SIGNMASK(a) (-((a) < 0))
# define OD_FLIPSIGNI(a, b) (((a) + OD_SIGNMASK(b)) ^ OD_SIGNMASK(b))

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_DSP_ODINTRIN_H_
