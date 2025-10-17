/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#ifndef RTC_BASE_SYSTEM_RTC_EXPORT_H_
#define RTC_BASE_SYSTEM_RTC_EXPORT_H_

// RTC_EXPORT is used to mark symbols as exported or imported when WebRTC is
// built or used as a shared library.
// When WebRTC is built as a static library the RTC_EXPORT macro expands to
// nothing.

#ifdef WEBRTC_ENABLE_SYMBOL_EXPORT

#ifdef WEBRTC_WIN

#ifdef WEBRTC_LIBRARY_IMPL
#define RTC_EXPORT __declspec(dllexport)
#else
#define RTC_EXPORT __declspec(dllimport)
#endif

#else  // WEBRTC_WIN

#if __has_attribute(visibility) && defined(WEBRTC_LIBRARY_IMPL)
#define RTC_EXPORT __attribute__((visibility("default")))
#endif

#endif  // WEBRTC_WIN

#endif  // WEBRTC_ENABLE_SYMBOL_EXPORT

#ifndef RTC_EXPORT
#define RTC_EXPORT
#endif

#endif  // RTC_BASE_SYSTEM_RTC_EXPORT_H_
