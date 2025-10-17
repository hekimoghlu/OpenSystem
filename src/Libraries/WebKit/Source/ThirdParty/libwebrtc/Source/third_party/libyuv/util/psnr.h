/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
// Get PSNR for video sequence. Assuming RAW 4:2:0 Y:Cb:Cr format

#ifndef UTIL_PSNR_H_  // NOLINT
#define UTIL_PSNR_H_

#include <math.h>  // For log10()

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(INT_TYPES_DEFINED) && !defined(UINT8_TYPE_DEFINED)
typedef unsigned char uint8_t;
#define UINT8_TYPE_DEFINED
#endif

static const double kMaxPSNR = 128.0;

// libyuv provides this function when linking library for jpeg support.
// TODO(fbarchard): make psnr lib compatible subset of libyuv.
#if !defined(HAVE_JPEG)
// Computer Sum of Squared Error (SSE).
// Pass this to ComputePSNR for final result.
double ComputeSumSquareError(const uint8_t* src_a,
                             const uint8_t* src_b,
                             int count);
#endif

// PSNR formula: psnr = 10 * log10 (Peak Signal^2 * size / sse)
// Returns 128.0 (kMaxPSNR) if sse is 0 (perfect match).
double ComputePSNR(double sse, double size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // UTIL_PSNR_H_  // NOLINT
