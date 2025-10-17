/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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
#ifndef AOM_STATS_RATE_HIST_H_
#define AOM_STATS_RATE_HIST_H_

#include "aom/aom_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

struct rate_hist;

struct rate_hist *init_rate_histogram(const aom_codec_enc_cfg_t *cfg,
                                      const aom_rational_t *fps);

void destroy_rate_histogram(struct rate_hist *hist);

void update_rate_histogram(struct rate_hist *hist,
                           const aom_codec_enc_cfg_t *cfg,
                           const aom_codec_cx_pkt_t *pkt);

void show_q_histogram(const int counts[64], int max_buckets);

void show_rate_histogram(struct rate_hist *hist, const aom_codec_enc_cfg_t *cfg,
                         int max_buckets);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_STATS_RATE_HIST_H_
