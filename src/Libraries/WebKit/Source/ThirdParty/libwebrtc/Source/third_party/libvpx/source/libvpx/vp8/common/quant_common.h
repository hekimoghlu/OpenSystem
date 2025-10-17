/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
#ifndef VPX_VP8_COMMON_QUANT_COMMON_H_
#define VPX_VP8_COMMON_QUANT_COMMON_H_

#include "string.h"
#include "blockd.h"
#include "onyxc_int.h"

#ifdef __cplusplus
extern "C" {
#endif

extern int vp8_ac_yquant(int QIndex);
extern int vp8_dc_quant(int QIndex, int Delta);
extern int vp8_dc2quant(int QIndex, int Delta);
extern int vp8_ac2quant(int QIndex, int Delta);
extern int vp8_dc_uv_quant(int QIndex, int Delta);
extern int vp8_ac_uv_quant(int QIndex, int Delta);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_QUANT_COMMON_H_
