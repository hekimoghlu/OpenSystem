/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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
#ifndef AOM_AOM_PORTS_SANITIZER_H_
#define AOM_AOM_PORTS_SANITIZER_H_

// AddressSanitizer support.

// Define AOM_ADDRESS_SANITIZER if AddressSanitizer is used.
// Clang.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define AOM_ADDRESS_SANITIZER 1
#endif
#endif  // defined(__has_feature)
// GCC.
#if defined(__SANITIZE_ADDRESS__)
#define AOM_ADDRESS_SANITIZER 1
#endif  // defined(__SANITIZE_ADDRESS__)

// Define the macros for AddressSanitizer manual memory poisoning. See
// https://github.com/google/sanitizers/wiki/AddressSanitizerManualPoisoning.
#if defined(AOM_ADDRESS_SANITIZER)
#include <sanitizer/asan_interface.h>
#else
#define ASAN_POISON_MEMORY_REGION(addr, size) ((void)(addr), (void)(size))
#define ASAN_UNPOISON_MEMORY_REGION(addr, size) ((void)(addr), (void)(size))
#endif

#endif  // AOM_AOM_PORTS_SANITIZER_H_
