/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
#ifndef VPX_VP9_ENCODER_VP9_SUBEXP_H_
#define VPX_VP9_ENCODER_VP9_SUBEXP_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "vpx_dsp/prob.h"

struct vpx_writer;

void vp9_write_prob_diff_update(struct vpx_writer *w, vpx_prob newp,
                                vpx_prob oldp);

void vp9_cond_prob_diff_update(struct vpx_writer *w, vpx_prob *oldp,
                               const unsigned int ct[2]);

int64_t vp9_prob_diff_update_savings_search(const unsigned int *ct,
                                            vpx_prob oldp, vpx_prob *bestp,
                                            vpx_prob upd);

int64_t vp9_prob_diff_update_savings_search_model(const unsigned int *ct,
                                                  const vpx_prob oldp,
                                                  vpx_prob *bestp, vpx_prob upd,
                                                  int stepsize);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_SUBEXP_H_
