/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#include "modules/desktop_capture/differ_block.h"

#include <string.h>

#include "rtc_base/system/arch.h"
#include "system_wrappers/include/cpu_features_wrapper.h"

// This needs to be after rtc_base/system/arch.h which defines
// architecture macros.
#if defined(WEBRTC_ARCH_X86_FAMILY)
#include "modules/desktop_capture/differ_vector_sse2.h"
#endif

namespace webrtc {

namespace {

bool VectorDifference_C(const uint8_t* image1, const uint8_t* image2) {
  return memcmp(image1, image2, kBlockSize * kBytesPerPixel) != 0;
}

}  // namespace

bool VectorDifference(const uint8_t* image1, const uint8_t* image2) {
  static bool (*diff_proc)(const uint8_t*, const uint8_t*) = nullptr;

  if (!diff_proc) {
#if defined(WEBRTC_ARCH_X86_FAMILY)
    bool have_sse2 = GetCPUInfo(kSSE2) != 0;
    // For x86 processors, check if SSE2 is supported.
    if (have_sse2 && kBlockSize == 32) {
      diff_proc = &VectorDifference_SSE2_W32;
    } else if (have_sse2 && kBlockSize == 16) {
      diff_proc = &VectorDifference_SSE2_W16;
    } else {
      diff_proc = &VectorDifference_C;
    }
#else
    // For other processors, always use C version.
    // TODO(hclam): Implement a NEON version.
    diff_proc = &VectorDifference_C;
#endif
  }

  return diff_proc(image1, image2);
}

bool BlockDifference(const uint8_t* image1,
                     const uint8_t* image2,
                     int height,
                     int stride) {
  for (int i = 0; i < height; i++) {
    if (VectorDifference(image1, image2)) {
      return true;
    }
    image1 += stride;
    image2 += stride;
  }
  return false;
}

bool BlockDifference(const uint8_t* image1, const uint8_t* image2, int stride) {
  return BlockDifference(image1, image2, kBlockSize, stride);
}

}  // namespace webrtc
