/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
#ifndef VPX_VP8_DECODER_EC_TYPES_H_
#define VPX_VP8_DECODER_EC_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_OVERLAPS 16

/* The area (pixel area in Q6) the block pointed to by bmi overlaps
 * another block with.
 */
typedef struct {
  int overlap;
  union b_mode_info *bmi;
} OVERLAP_NODE;

/* Structure to keep track of overlapping blocks on a block level. */
typedef struct {
  /* TODO(holmer): This array should be exchanged for a linked list */
  OVERLAP_NODE overlaps[MAX_OVERLAPS];
} B_OVERLAP;

/* Structure used to hold all the overlaps of a macroblock. The overlaps of a
 * macroblock is further divided into block overlaps.
 */
typedef struct {
  B_OVERLAP overlaps[16];
} MB_OVERLAP;

/* Structure for keeping track of motion vectors and which reference frame they
 * refer to. Used for motion vector interpolation.
 */
typedef struct {
  MV mv;
  MV_REFERENCE_FRAME ref_frame;
} EC_BLOCK;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_DECODER_EC_TYPES_H_
