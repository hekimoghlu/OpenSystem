/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
#ifndef VPX_VP9_COMMON_VP9_MFQE_H_
#define VPX_VP9_COMMON_VP9_MFQE_H_

#ifdef __cplusplus
extern "C" {
#endif

// Multiframe Quality Enhancement.
// The aim for MFQE is to replace pixel blocks in the current frame with
// the correlated pixel blocks (with higher quality) in the last frame.
// The replacement can only be taken in stationary blocks by checking
// the motion of the blocks and other conditions such as the SAD of
// the current block and correlated block, the variance of the block
// difference, etc.
void vp9_mfqe(struct VP9Common *cm);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_MFQE_H_
