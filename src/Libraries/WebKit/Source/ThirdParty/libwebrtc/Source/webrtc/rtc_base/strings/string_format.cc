/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
#include "rtc_base/strings/string_format.h"

#include <cstdarg>

#include "rtc_base/checks.h"

namespace rtc {

namespace {

// This is an arbitrary limitation that can be changed if necessary, or removed
// if someone has the time and inclination to replicate the fancy logic from
// Chromium's base::StringPrinf().
constexpr int kMaxSize = 512;

}  // namespace

std::string StringFormat(const char* fmt, ...) {
  char buffer[kMaxSize];
  va_list args;
  va_start(args, fmt);
  int result = vsnprintf(buffer, kMaxSize, fmt, args);
  va_end(args);
  RTC_DCHECK_GE(result, 0) << "ERROR: vsnprintf() failed with error " << result;
  RTC_DCHECK_LT(result, kMaxSize)
      << "WARNING: string was truncated from " << result << " to "
      << (kMaxSize - 1) << " characters";
  return std::string(buffer);
}

}  // namespace rtc
