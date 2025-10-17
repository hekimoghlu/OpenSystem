/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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
#ifndef RTC_BASE_SYSTEM_NO_UNIQUE_ADDRESS_H_
#define RTC_BASE_SYSTEM_NO_UNIQUE_ADDRESS_H_

// RTC_NO_UNIQUE_ADDRESS is a portable annotation to tell the compiler that
// a data member need not have an address distinct from all other non-static
// data members of its class.
// It allows empty types to actually occupy zero bytes as class members,
// instead of occupying at least one byte just so that they get their own
// address. There is almost never any reason not to use it on class members
// that could possibly be empty.
// The macro expands to [[no_unique_address]] if the compiler supports the
// attribute, it expands to nothing otherwise.
// Clang should supports this attribute since C++11, while other compilers
// should add support for it starting from C++20. Among clang compilers,
// clang-cl doesn't support it yet and support is unclear also when the target
// platform is iOS.
#ifndef __has_cpp_attribute
#define __has_cpp_attribute(x) 0
#endif
#if __has_cpp_attribute(no_unique_address)
// NOLINTNEXTLINE(whitespace/braces)
#define RTC_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#define RTC_NO_UNIQUE_ADDRESS
#endif

#endif  // RTC_BASE_SYSTEM_NO_UNIQUE_ADDRESS_H_
