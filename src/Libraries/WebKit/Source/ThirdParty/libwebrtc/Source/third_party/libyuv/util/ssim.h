/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
// Get SSIM for video sequence. Assuming RAW 4:2:0 Y:Cb:Cr format

#ifndef UTIL_SSIM_H_
#define UTIL_SSIM_H_

#include <math.h>  // For log10()

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(INT_TYPES_DEFINED) && !defined(UINT8_TYPE_DEFINED)
typedef unsigned char uint8_t;
#define UINT8_TYPE_DEFINED
#endif

double CalcSSIM(const uint8_t* org,
                const uint8_t* rec,
                const int image_width,
                const int image_height);

double CalcLSSIM(double ssim);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // UTIL_SSIM_H_
