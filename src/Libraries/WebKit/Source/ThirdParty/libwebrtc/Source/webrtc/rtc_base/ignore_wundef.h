/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
#ifndef RTC_BASE_IGNORE_WUNDEF_H_
#define RTC_BASE_IGNORE_WUNDEF_H_

// If a header file uses #if on possibly undefined macros (and it's for some
// reason not possible to just fix the header file), include it like this:
//
//   RTC_PUSH_IGNORING_WUNDEF()
//   #include "misbehaving_header.h"
//   RTC_POP_IGNORING_WUNDEF()
//
// This will cause the compiler to not emit -Wundef warnings for that file.

#ifdef __clang__
#define RTC_PUSH_IGNORING_WUNDEF() \
  _Pragma("clang diagnostic push") \
      _Pragma("clang diagnostic ignored \"-Wundef\"")
#define RTC_POP_IGNORING_WUNDEF() _Pragma("clang diagnostic pop")
#else
#define RTC_PUSH_IGNORING_WUNDEF()
#define RTC_POP_IGNORING_WUNDEF()
#endif  // __clang__

#endif  // RTC_BASE_IGNORE_WUNDEF_H_
