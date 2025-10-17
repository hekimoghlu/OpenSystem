/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
#ifndef MODULES_DESKTOP_CAPTURE_DIFFER_BLOCK_H_
#define MODULES_DESKTOP_CAPTURE_DIFFER_BLOCK_H_

#include <stdint.h>

namespace webrtc {

// Size (in pixels) of each square block used for diffing. This must be a
// multiple of sizeof(uint64)/8.
const int kBlockSize = 32;

// Format: BGRA 32 bit.
const int kBytesPerPixel = 4;

// Low level function to compare 2 vectors of pixels of size kBlockSize. Returns
// whether the blocks differ.
bool VectorDifference(const uint8_t* image1, const uint8_t* image2);

// Low level function to compare 2 blocks of pixels of size
// (kBlockSize, `height`).  Returns whether the blocks differ.
bool BlockDifference(const uint8_t* image1,
                     const uint8_t* image2,
                     int height,
                     int stride);

// Low level function to compare 2 blocks of pixels of size
// (kBlockSize, kBlockSize).  Returns whether the blocks differ.
bool BlockDifference(const uint8_t* image1, const uint8_t* image2, int stride);

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_DIFFER_BLOCK_H_
