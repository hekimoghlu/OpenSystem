/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#ifndef VPX_VP9_ENCODER_VP9_TPL_MODEL_H_
#define VPX_VP9_ENCODER_VP9_TPL_MODEL_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_LOG2_E
#define M_LOG2_E 0.693147180559945309417
#endif
#define log2f(x) (log(x) / (float)M_LOG2_E)

#define TPL_DEP_COST_SCALE_LOG2 4

typedef struct GF_PICTURE {
  YV12_BUFFER_CONFIG *frame;
  int ref_frame[3];
  FRAME_UPDATE_TYPE update_type;
} GF_PICTURE;

void vp9_init_tpl_buffer(VP9_COMP *cpi);
void vp9_setup_tpl_stats(VP9_COMP *cpi);
void vp9_free_tpl_buffer(VP9_COMP *cpi);
void vp9_estimate_tpl_qp_gop(VP9_COMP *cpi);

void vp9_wht_fwd_txfm(int16_t *src_diff, int bw, tran_low_t *coeff,
                      TX_SIZE tx_size);
#if CONFIG_VP9_HIGHBITDEPTH
void vp9_highbd_wht_fwd_txfm(int16_t *src_diff, int bw, tran_low_t *coeff,
                             TX_SIZE tx_size);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_TPL_MODEL_H_
