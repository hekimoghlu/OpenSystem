/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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
#ifndef TRIO_NAN_H
#define TRIO_NAN_H

#include "triodef.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  TRIO_FP_INFINITE,
  TRIO_FP_NAN,
  TRIO_FP_NORMAL,
  TRIO_FP_SUBNORMAL,
  TRIO_FP_ZERO
};

/*
 * Return NaN (Not-a-Number).
 */
TRIO_PUBLIC double trio_nan TRIO_PROTO((void));

/*
 * Return positive infinity.
 */
TRIO_PUBLIC double trio_pinf TRIO_PROTO((void));

/*
 * Return negative infinity.
 */
TRIO_PUBLIC double trio_ninf TRIO_PROTO((void));

/*
 * Return negative zero.
 */
TRIO_PUBLIC double trio_nzero TRIO_PROTO((TRIO_NOARGS));

/*
 * If number is a NaN return non-zero, otherwise return zero.
 */
TRIO_PUBLIC int trio_isnan TRIO_PROTO((double number));

/*
 * If number is positive infinity return 1, if number is negative
 * infinity return -1, otherwise return 0.
 */
TRIO_PUBLIC int trio_isinf TRIO_PROTO((double number));

/*
 * If number is finite return non-zero, otherwise return zero.
 */
#if 0
	/* Temporary fix - these 2 routines not used in libxml */
TRIO_PUBLIC int trio_isfinite TRIO_PROTO((double number));

TRIO_PUBLIC int trio_fpclassify TRIO_PROTO((double number));
#endif

TRIO_PUBLIC int trio_signbit TRIO_PROTO((double number));

TRIO_PUBLIC int trio_fpclassify_and_signbit TRIO_PROTO((double number, int *is_negative));

#ifdef __cplusplus
}
#endif

#endif /* TRIO_NAN_H */
