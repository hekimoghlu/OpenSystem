/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
#ifndef VPX_VP9_COMMON_VP9_PPFLAGS_H_
#define VPX_VP9_COMMON_VP9_PPFLAGS_H_

#ifdef __cplusplus
extern "C" {
#endif

enum {
  VP9D_NOFILTERING = 0,
  VP9D_DEBLOCK = 1 << 0,
  VP9D_DEMACROBLOCK = 1 << 1,
  VP9D_ADDNOISE = 1 << 2,
  VP9D_MFQE = 1 << 3
};

typedef struct {
  int post_proc_flag;
  int deblocking_level;
  int noise_level;
} vp9_ppflags_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_PPFLAGS_H_
