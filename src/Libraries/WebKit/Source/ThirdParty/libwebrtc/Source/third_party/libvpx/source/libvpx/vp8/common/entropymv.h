/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#ifndef VPX_VP8_COMMON_ENTROPYMV_H_
#define VPX_VP8_COMMON_ENTROPYMV_H_

#include "treecoder.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  mv_max = 1023,             /* max absolute value of a MV component */
  MVvals = (2 * mv_max) + 1, /* # possible values "" */
  mvfp_max = 255, /* max absolute value of a full pixel MV component */
  MVfpvals = (2 * mvfp_max) + 1, /* # possible full pixel MV values */

  mvlong_width = 10, /* Large MVs have 9 bit magnitudes */
  mvnum_short = 8,   /* magnitudes 0 through 7 */

  /* probability offsets for coding each MV component */

  mvpis_short = 0, /* short (<= 7) vs long (>= 8) */
  MVPsign,         /* sign for non-zero */
  MVPshort,        /* 8 short values = 7-position tree */

  MVPbits = MVPshort + mvnum_short - 1, /* mvlong_width long value bits */
  MVPcount = MVPbits + mvlong_width     /* (with independent probabilities) */
};

typedef struct mv_context {
  vp8_prob prob[MVPcount]; /* often come in row, col pairs */
} MV_CONTEXT;

extern const MV_CONTEXT vp8_mv_update_probs[2], vp8_default_mv_context[2];

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_ENTROPYMV_H_
