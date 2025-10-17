/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
#ifndef VPX_VP8_DECODER_DECODERTHREADING_H_
#define VPX_VP8_DECODER_DECODERTHREADING_H_

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_MULTITHREAD
int vp8mt_decode_mb_rows(VP8D_COMP *pbi, MACROBLOCKD *xd);
void vp8_decoder_remove_threads(VP8D_COMP *pbi);
void vp8_decoder_create_threads(VP8D_COMP *pbi);
void vp8mt_alloc_temp_buffers(VP8D_COMP *pbi, int width, int prev_mb_rows);
void vp8mt_de_alloc_temp_buffers(VP8D_COMP *pbi, int mb_rows);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_DECODER_DECODERTHREADING_H_
