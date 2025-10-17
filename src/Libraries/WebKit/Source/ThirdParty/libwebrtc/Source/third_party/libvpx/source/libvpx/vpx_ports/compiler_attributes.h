/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
#ifndef VPX_VPX_PORTS_COMPILER_ATTRIBUTES_H_
#define VPX_VPX_PORTS_COMPILER_ATTRIBUTES_H_

#if !defined(__has_feature)
#define __has_feature(x) 0
#endif  // !defined(__has_feature)

#if !defined(__has_attribute)
#define __has_attribute(x) 0
#endif  // !defined(__has_attribute)

//------------------------------------------------------------------------------
// Sanitizer attributes.

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#define VPX_WITH_ASAN 1
#else
#define VPX_WITH_ASAN 0
#endif  // __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)

#if defined(__clang__) && __has_attribute(no_sanitize)
// Both of these have defined behavior and are used in certain operations or
// optimizations thereof. There are cases where an overflow may be unintended,
// however, so use of these attributes should be done with care.
#define VPX_NO_UNSIGNED_OVERFLOW_CHECK \
  __attribute__((no_sanitize("unsigned-integer-overflow")))
#if __clang_major__ >= 12
#define VPX_NO_UNSIGNED_SHIFT_CHECK \
  __attribute__((no_sanitize("unsigned-shift-base")))
#endif  // __clang__ >= 12
#endif  // __clang__

#ifndef VPX_NO_UNSIGNED_OVERFLOW_CHECK
#define VPX_NO_UNSIGNED_OVERFLOW_CHECK
#endif
#ifndef VPX_NO_UNSIGNED_SHIFT_CHECK
#define VPX_NO_UNSIGNED_SHIFT_CHECK
#endif

//------------------------------------------------------------------------------
// Variable attributes.

#if __has_attribute(uninitialized)
// Attribute "uninitialized" disables -ftrivial-auto-var-init=pattern for
// the specified variable.
//
// -ftrivial-auto-var-init is security risk mitigation feature, so attribute
// should not be used "just in case", but only to fix real performance
// bottlenecks when other approaches do not work. In general the compiler is
// quite effective at eliminating unneeded initializations introduced by the
// flag, e.g. when they are followed by actual initialization by a program.
// However if compiler optimization fails and code refactoring is hard, the
// attribute can be used as a workaround.
#define VPX_UNINITIALIZED __attribute__((uninitialized))
#else
#define VPX_UNINITIALIZED
#endif  // __has_attribute(uninitialized)

#endif  // VPX_VPX_PORTS_COMPILER_ATTRIBUTES_H_
