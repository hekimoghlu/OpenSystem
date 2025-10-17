/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
#include "entropymv.h"

/* clang-format off */
const MV_CONTEXT vp8_mv_update_probs[2] = {
  { {
      237,
      246,
      253, 253, 254, 254, 254, 254, 254,
      254, 254, 254, 254, 254, 250, 250, 252, 254, 254
  } },
  { {
      231,
      243,
      245, 253, 254, 254, 254, 254, 254,
      254, 254, 254, 254, 254, 251, 251, 254, 254, 254
  } }
};
/* clang-format on */

const MV_CONTEXT vp8_default_mv_context[2] = {
  { {
      /* row */
      162,                                            /* is short */
      128,                                            /* sign */
      225, 146, 172, 147, 214, 39, 156,               /* short tree */
      128, 129, 132, 75, 145, 178, 206, 239, 254, 254 /* long bits */
  } },

  { {
      /* same for column */
      164,                                            /* is short */
      128,                                            /**/
      204, 170, 119, 235, 140, 230, 228,              /**/
      128, 130, 130, 74, 148, 180, 203, 236, 254, 254 /* long bits */

  } }
};
