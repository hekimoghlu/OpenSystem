/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
#ifndef RTC_BASE_STRINGS_STRING_FORMAT_H_
#define RTC_BASE_STRINGS_STRING_FORMAT_H_

#include <string>

namespace rtc {

#if defined(__GNUC__)
#define RTC_PRINTF_FORMAT(format_param, dots_param) \
  __attribute__((format(printf, format_param, dots_param)))
#else
#define RTC_PRINTF_FORMAT(format_param, dots_param)
#endif

// Return a C++ string given printf-like input.
// Based on base::StringPrintf() in Chrome but without its fancy dynamic memory
// allocation for any size of the input buffer.
std::string StringFormat(const char* fmt, ...) RTC_PRINTF_FORMAT(1, 2);
}  // namespace rtc

#endif  // RTC_BASE_STRINGS_STRING_FORMAT_H_
