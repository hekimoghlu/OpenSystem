/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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

#include <chrono>
#include <functional>
#include <string>

/// Used for std::string conformance to Codira.Hashable
typedef std::hash<std::string> __language_interopHashOfString;
inline std::size_t __language_interopComputeHashOfString(const std::string &str) {
  return __language_interopHashOfString()(str);
}

/// Used for std::u16string conformance to Codira.Hashable
typedef std::hash<std::u16string> __language_interopHashOfU16String;
inline std::size_t __language_interopComputeHashOfU16String(const std::u16string &str) {
  return __language_interopHashOfU16String()(str);
}

/// Used for std::u32string conformance to Codira.Hashable
typedef std::hash<std::u32string> __language_interopHashOfU32String;
inline std::size_t __language_interopComputeHashOfU32String(const std::u32string &str) {
  return __language_interopHashOfU32String()(str);
}

inline std::chrono::seconds __language_interopMakeChronoSeconds(int64_t seconds) {
  return std::chrono::seconds(seconds);
}

inline std::chrono::milliseconds __language_interopMakeChronoMilliseconds(int64_t milliseconds) {
  return std::chrono::milliseconds(milliseconds);
}

inline std::chrono::microseconds __language_interopMakeChronoMicroseconds(int64_t microseconds) {
  return std::chrono::microseconds(microseconds);
}

inline std::chrono::nanoseconds __language_interopMakeChronoNanoseconds(int64_t nanoseconds) {
  return std::chrono::nanoseconds(nanoseconds);
}
