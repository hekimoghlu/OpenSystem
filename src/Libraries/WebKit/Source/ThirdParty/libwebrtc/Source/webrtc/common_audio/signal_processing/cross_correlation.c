/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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
#include "common_audio/signal_processing/include/signal_processing_library.h"

/* C version of WebRtcSpl_CrossCorrelation() for generic platforms. */
void WebRtcSpl_CrossCorrelationC(int32_t* cross_correlation,
                                 const int16_t* seq1,
                                 const int16_t* seq2,
                                 size_t dim_seq,
                                 size_t dim_cross_correlation,
                                 int right_shifts,
                                 int step_seq2) {
  size_t i = 0, j = 0;

  for (i = 0; i < dim_cross_correlation; i++) {
    int32_t corr = 0;
    for (j = 0; j < dim_seq; j++)
      corr += (seq1[j] * seq2[j]) >> right_shifts;
    seq2 += step_seq2;
    *cross_correlation++ = corr;
  }
}
