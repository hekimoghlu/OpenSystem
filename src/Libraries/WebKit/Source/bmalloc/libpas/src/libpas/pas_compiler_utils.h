/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
#pragma once

#if defined(__clang__)
#define PAS_COMPILER_CLANG 1
#endif

/* PAS_ALLOW_UNSAFE_BUFFER_USAGE */
#if PAS_COMPILER_CLANG
#define PAS_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Wunsafe-buffer-usage\"")

#define PAS_ALLOW_UNSAFE_BUFFER_USAGE_END \
    _Pragma("clang diagnostic pop")
#else
#define PAS_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
#define PAS_ALLOW_UNSAFE_BUFFER_USAGE_END
#endif

/* PAS_UNSAFE_BUFFER_USAGE */
#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#ifndef __has_cpp_attribute
#define __has_cpp_attribute(x) 0
#endif

#if PAS_COMPILER_CLANG
#if __has_cpp_attribute(clang::unsafe_buffer_usage)
#define PAS_UNSAFE_BUFFER_USAGE [[clang::unsafe_buffer_usage]]
#elif __has_attribute(unsafe_buffer_usage)
#define PAS_UNSAFE_BUFFER_USAGE __attribute__((__unsafe_buffer_usage__))
#else
#define PAS_UNSAFE_BUFFER_USAGE
#endif
#else
#define PAS_UNSAFE_BUFFER_USAGE
#endif
