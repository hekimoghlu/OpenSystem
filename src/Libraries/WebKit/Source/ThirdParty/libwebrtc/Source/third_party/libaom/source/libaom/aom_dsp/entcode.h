/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
#ifndef AOM_AOM_DSP_ENTCODE_H_
#define AOM_AOM_DSP_ENTCODE_H_

#include <limits.h>
#include <stddef.h>
#include "aom_dsp/odintrin.h"
#include "aom_dsp/prob.h"

#define EC_PROB_SHIFT 6
#define EC_MIN_PROB 4  // must be <= (1<<EC_PROB_SHIFT)/16

/*OPT: od_ec_window must be at least 32 bits, but if you have fast arithmetic
   on a larger type, you can speed up the decoder by using it here.*/
typedef uint32_t od_ec_window;

/*The size in bits of od_ec_window.*/
#define OD_EC_WINDOW_SIZE ((int)sizeof(od_ec_window) * CHAR_BIT)

/*The resolution of fractional-precision bit usage measurements, i.e.,
   3 => 1/8th bits.*/
#define OD_BITRES (3)

#define OD_ICDF AOM_ICDF

/*See entcode.c for further documentation.*/

OD_WARN_UNUSED_RESULT uint32_t od_ec_tell_frac(uint32_t nbits_total,
                                               uint32_t rng);

#endif  // AOM_AOM_DSP_ENTCODE_H_
