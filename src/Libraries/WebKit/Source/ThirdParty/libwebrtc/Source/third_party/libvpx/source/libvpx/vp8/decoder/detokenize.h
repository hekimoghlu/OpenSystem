/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
#ifndef VPX_VP8_DECODER_DETOKENIZE_H_
#define VPX_VP8_DECODER_DETOKENIZE_H_

#include "onyxd_int.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp8_reset_mb_tokens_context(MACROBLOCKD *x);
int vp8_decode_mb_tokens(VP8D_COMP *, MACROBLOCKD *);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_DECODER_DETOKENIZE_H_
