/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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

// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#ifndef LIBWEBM_COMMON_WEBM_ENDIAN_H_
#define LIBWEBM_COMMON_WEBM_ENDIAN_H_

#include <stdint.h>

namespace libwebm {

// Swaps unsigned 32 bit values to big endian if needed. Returns |value| if
// architecture is big endian. Returns big endian value if architecture is
// little endian. Returns 0 otherwise.
uint32_t host_to_bigendian(uint32_t value);

// Swaps unsigned 32 bit values to little endian if needed. Returns |value| if
// architecture is big endian. Returns little endian value if architecture is
// little endian. Returns 0 otherwise.
uint32_t bigendian_to_host(uint32_t value);

// Swaps unsigned 64 bit values to big endian if needed. Returns |value| if
// architecture is big endian. Returns big endian value if architecture is
// little endian. Returns 0 otherwise.
uint64_t host_to_bigendian(uint64_t value);

// Swaps unsigned 64 bit values to little endian if needed. Returns |value| if
// architecture is big endian. Returns little endian value if architecture is
// little endian. Returns 0 otherwise.
uint64_t bigendian_to_host(uint64_t value);

}  // namespace libwebm

#endif  // LIBWEBM_COMMON_WEBM_ENDIAN_H_
