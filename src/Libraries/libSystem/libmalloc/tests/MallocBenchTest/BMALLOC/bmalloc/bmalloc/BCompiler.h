/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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

/* BCOMPILER() - the compiler being used to build the project */
#define BCOMPILER(BFEATURE) (defined BCOMPILER_##BFEATURE && BCOMPILER_##BFEATURE)

/* BCOMPILER_HAS_CLANG_FEATURE() - whether the compiler supports a particular language or library feature. */
/* http://clang.llvm.org/docs/LanguageExtensions.html#has-feature-and-has-extension */
#ifdef __has_feature
#define BCOMPILER_HAS_CLANG_FEATURE(x) __has_feature(x)
#else
#define BCOMPILER_HAS_CLANG_FEATURE(x) 0
#endif

#define BASAN_ENABLED BCOMPILER_HAS_CLANG_FEATURE(address_sanitizer)

/* BCOMPILER(GCC_COMPATIBLE) - GNU Compiler Collection or compatibles */

#if defined(__GNUC__)
#define BCOMPILER_GCC_COMPATIBLE 1
#endif

/* BNO_RETURN */

#if !defined(BNO_RETURN) && BCOMPILER(GCC_COMPATIBLE)
#define BNO_RETURN __attribute((__noreturn__))
#endif

#if !defined(BNO_RETURN)
#define BNO_RETURN
#endif

