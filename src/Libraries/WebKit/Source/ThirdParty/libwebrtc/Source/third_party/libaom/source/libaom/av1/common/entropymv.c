/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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
#include "av1/common/av1_common_int.h"
#include "av1/common/entropymv.h"

static const nmv_context default_nmv_context = {
  { AOM_CDF4(4096, 11264, 19328) },  // joints_cdf
  { {
        // Vertical component
        { AOM_CDF11(28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757,
                    32762, 32767) },  // class_cdf // fp
        { { AOM_CDF4(16384, 24576, 26624) },
          { AOM_CDF4(12288, 21248, 24128) } },  // class0_fp_cdf
        { AOM_CDF4(8192, 17408, 21248) },       // fp_cdf
        { AOM_CDF2(128 * 128) },                // sign_cdf
        { AOM_CDF2(160 * 128) },                // class0_hp_cdf
        { AOM_CDF2(128 * 128) },                // hp_cdf
        { AOM_CDF2(216 * 128) },                // class0_cdf
        { { AOM_CDF2(128 * 136) },
          { AOM_CDF2(128 * 140) },
          { AOM_CDF2(128 * 148) },
          { AOM_CDF2(128 * 160) },
          { AOM_CDF2(128 * 176) },
          { AOM_CDF2(128 * 192) },
          { AOM_CDF2(128 * 224) },
          { AOM_CDF2(128 * 234) },
          { AOM_CDF2(128 * 234) },
          { AOM_CDF2(128 * 240) } },  // bits_cdf
    },
    {
        // Horizontal component
        { AOM_CDF11(28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757,
                    32762, 32767) },  // class_cdf // fp
        { { AOM_CDF4(16384, 24576, 26624) },
          { AOM_CDF4(12288, 21248, 24128) } },  // class0_fp_cdf
        { AOM_CDF4(8192, 17408, 21248) },       // fp_cdf
        { AOM_CDF2(128 * 128) },                // sign_cdf
        { AOM_CDF2(160 * 128) },                // class0_hp_cdf
        { AOM_CDF2(128 * 128) },                // hp_cdf
        { AOM_CDF2(216 * 128) },                // class0_cdf
        { { AOM_CDF2(128 * 136) },
          { AOM_CDF2(128 * 140) },
          { AOM_CDF2(128 * 148) },
          { AOM_CDF2(128 * 160) },
          { AOM_CDF2(128 * 176) },
          { AOM_CDF2(128 * 192) },
          { AOM_CDF2(128 * 224) },
          { AOM_CDF2(128 * 234) },
          { AOM_CDF2(128 * 234) },
          { AOM_CDF2(128 * 240) } },  // bits_cdf
    } },
};

void av1_init_mv_probs(AV1_COMMON *cm) {
  // NB: this sets CDFs too
  cm->fc->nmvc = default_nmv_context;
  cm->fc->ndvc = default_nmv_context;
}
