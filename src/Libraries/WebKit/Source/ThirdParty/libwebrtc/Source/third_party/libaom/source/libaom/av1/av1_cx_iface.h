/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
#ifndef AOM_AV1_AV1_CX_IFACE_H_
#define AOM_AV1_AV1_CX_IFACE_H_
#include "av1/encoder/encoder.h"
#include "aom/aom_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

AV1EncoderConfig av1_get_encoder_config(const aom_codec_enc_cfg_t *cfg);

aom_codec_err_t av1_create_context_and_bufferpool(AV1_PRIMARY *ppi,
                                                  AV1_COMP **p_cpi,
                                                  BufferPool **p_buffer_pool,
                                                  const AV1EncoderConfig *oxcf,
                                                  COMPRESSOR_STAGE stage,
                                                  int lap_lag_in_frames);

void av1_destroy_context_and_bufferpool(AV1_COMP *cpi,
                                        BufferPool **p_buffer_pool);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_AV1_CX_IFACE_H_
