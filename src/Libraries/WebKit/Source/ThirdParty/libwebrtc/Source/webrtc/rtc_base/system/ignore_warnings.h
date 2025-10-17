/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#ifndef RTC_BASE_SYSTEM_IGNORE_WARNINGS_H_
#define RTC_BASE_SYSTEM_IGNORE_WARNINGS_H_

#ifdef __clang__
#define RTC_PUSH_IGNORING_WFRAME_LARGER_THAN() \
  _Pragma("clang diagnostic push")             \
      _Pragma("clang diagnostic ignored \"-Wframe-larger-than=\"")
#define RTC_POP_IGNORING_WFRAME_LARGER_THAN() _Pragma("clang diagnostic pop")
#elif __GNUC__
#define RTC_PUSH_IGNORING_WFRAME_LARGER_THAN() \
  _Pragma("GCC diagnostic push")               \
      _Pragma("GCC diagnostic ignored \"-Wframe-larger-than=\"")
#define RTC_POP_IGNORING_WFRAME_LARGER_THAN() _Pragma("GCC diagnostic pop")
#else
#define RTC_PUSH_IGNORING_WFRAME_LARGER_THAN()
#define RTC_POP_IGNORING_WFRAME_LARGER_THAN()
#endif

#endif  // RTC_BASE_SYSTEM_IGNORE_WARNINGS_H_
