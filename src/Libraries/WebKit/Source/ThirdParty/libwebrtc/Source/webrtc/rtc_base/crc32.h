/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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
#ifndef RTC_BASE_CRC32_H_
#define RTC_BASE_CRC32_H_

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "absl/strings/string_view.h"

namespace rtc {

// Updates a CRC32 checksum with `len` bytes from `buf`. `initial` holds the
// checksum result from the previous update; for the first call, it should be 0.
uint32_t UpdateCrc32(uint32_t initial, const void* buf, size_t len);

// Computes a CRC32 checksum using `len` bytes from `buf`.
inline uint32_t ComputeCrc32(const void* buf, size_t len) {
  return UpdateCrc32(0, buf, len);
}
inline uint32_t ComputeCrc32(absl::string_view str) {
  return ComputeCrc32(str.data(), str.size());
}

}  // namespace rtc

#endif  // RTC_BASE_CRC32_H_
