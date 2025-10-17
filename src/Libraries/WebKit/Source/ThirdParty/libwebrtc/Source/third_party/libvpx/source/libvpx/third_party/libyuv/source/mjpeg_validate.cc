/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#include "libyuv/mjpeg_decoder.h"

#include <string.h>  // For memchr.

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Helper function to scan for EOI marker (0xff 0xd9).
static LIBYUV_BOOL ScanEOI(const uint8_t* sample, size_t sample_size) {
  if (sample_size >= 2) {
    const uint8_t* end = sample + sample_size - 1;
    const uint8_t* it = sample;
    while (it < end) {
      // TODO(fbarchard): scan for 0xd9 instead.
      it = (const uint8_t*)(memchr(it, 0xff, end - it));
      if (it == NULL) {
        break;
      }
      if (it[1] == 0xd9) {
        return LIBYUV_TRUE;  // Success: Valid jpeg.
      }
      ++it;  // Skip over current 0xff.
    }
  }
  // ERROR: Invalid jpeg end code not found. Size sample_size
  return LIBYUV_FALSE;
}

// Helper function to validate the jpeg appears intact.
LIBYUV_BOOL ValidateJpeg(const uint8_t* sample, size_t sample_size) {
  // Maximum size that ValidateJpeg will consider valid.
  const size_t kMaxJpegSize = 0x7fffffffull;
  const size_t kBackSearchSize = 1024;
  if (sample_size < 64 || sample_size > kMaxJpegSize || !sample) {
    // ERROR: Invalid jpeg size: sample_size
    return LIBYUV_FALSE;
  }
  if (sample[0] != 0xff || sample[1] != 0xd8) {  // SOI marker
    // ERROR: Invalid jpeg initial start code
    return LIBYUV_FALSE;
  }

  // Look for the End Of Image (EOI) marker near the end of the buffer.
  if (sample_size > kBackSearchSize) {
    if (ScanEOI(sample + sample_size - kBackSearchSize, kBackSearchSize)) {
      return LIBYUV_TRUE;  // Success: Valid jpeg.
    }
    // Reduce search size for forward search.
    sample_size = sample_size - kBackSearchSize + 1;
  }
  // Step over SOI marker and scan for EOI.
  return ScanEOI(sample + 2, sample_size - 2);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
