/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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
#ifndef VPX_VP8_ENCODER_TOKENIZE_H_
#define VPX_VP8_ENCODER_TOKENIZE_H_

#include "vp8/common/entropy.h"
#include "block.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  short Token;
  short Extra;
} TOKENVALUE;

typedef struct {
  const vp8_prob *context_tree;
  short Extra;
  unsigned char Token;
  unsigned char skip_eob_node;
} TOKENEXTRA;

int rd_cost_mby(MACROBLOCKD *);

extern const short *const vp8_dct_value_cost_ptr;
/* TODO: The Token field should be broken out into a separate char array to
 *  improve cache locality, since it's needed for costing when the rest of the
 *  fields are not.
 */
extern const TOKENVALUE *const vp8_dct_value_tokens_ptr;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_TOKENIZE_H_
