/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#ifndef VPX_TEST_TEST_VECTORS_H_
#define VPX_TEST_TEST_VECTORS_H_

#include "./vpx_config.h"

namespace libvpx_test {

#if CONFIG_VP8_DECODER
extern const int kNumVP8TestVectors;
extern const char *const kVP8TestVectors[];
#endif

#if CONFIG_VP9_DECODER
extern const int kNumVP9TestVectors;
extern const char *const kVP9TestVectors[];
extern const int kNumVP9TestVectorsSvc;
extern const char *const kVP9TestVectorsSvc[];
extern const int kNumVP9TestVectorsResize;
extern const char *const kVP9TestVectorsResize[];
#endif  // CONFIG_VP9_DECODER

}  // namespace libvpx_test

#endif  // VPX_TEST_TEST_VECTORS_H_
