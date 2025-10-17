/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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

/* BCOMPILER_HAS_CLANG_DECLSPEC() - whether the compiler supports a Microsoft style __declspec attribute. */
/* https://clang.llvm.org/docs/LanguageExtensions.html#has-declspec-attribute */
#ifdef __has_declspec_attribute
#define BCOMPILER_HAS_CLANG_DECLSPEC(x) __has_declspec_attribute(x)
#else
#define BCOMPILER_HAS_CLANG_DECLSPEC(x) 0
#endif

#if defined(__clang__)
#define BCOMPILER_CLANG 1
#endif

/* BCOMPILER(GCC_COMPATIBLE) - GNU Compiler Collection or compatibles */

#if defined(__GNUC__)
#define BCOMPILER_GCC_COMPATIBLE 1
#endif

#if defined(_MSC_VER)
#define BCOMPILER_MSVC 1
#if _MSC_VER < 1910
#error "Please use a newer version of Visual Studio. WebKit requires VS2017 or newer to compile."
#endif
#endif

/* BNO_RETURN */

#if !defined(BNO_RETURN)
#if BCOMPILER(GCC_COMPATIBLE)
#define BNO_RETURN __attribute((__noreturn__))
#elif BCOMPILER(MSVC)
#define BNO_RETURN __declspec(noreturn)
#else
#define BNO_RETURN
#endif
#endif

/* BFALLTHROUGH */

#if !defined(BFALLTHROUGH) && defined(__cplusplus) && defined(__has_cpp_attribute)

#if __has_cpp_attribute(fallthrough)
#define BFALLTHROUGH [[fallthrough]]
#elif __has_cpp_attribute(clang::fallthrough)
#define BFALLTHROUGH [[clang::fallthrough]]
#elif __has_cpp_attribute(gnu::fallthrough)
#define BFALLTHROUGH [[gnu::fallthrough]]
#endif

#endif // !defined(BFALLTHROUGH) && defined(__cplusplus) && defined(__has_cpp_attribute)

#if !defined(BFALLTHROUGH)
#define BFALLTHROUGH
#endif

/* BLIKELY */

#if !defined(BLIKELY) && BCOMPILER(GCC_COMPATIBLE)
#define BLIKELY(x) __builtin_expect(!!(x), 1)
#endif

#if !defined(BLIKELY)
#define BLIKELY(x) (x)
#endif

/* BUNLIKELY */

#if !defined(BUNLIKELY) && BCOMPILER(GCC_COMPATIBLE)
#define BUNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

#if !defined(BUNLIKELY)
#define BUNLIKELY(x) (x)
#endif

/* BUNUSED_TYPE_ALIAS */

#if !defined(BUNUSED_TYPE_ALIAS) && BCOMPILER(GCC_COMPATIBLE)
#define BUNUSED_TYPE_ALIAS __attribute__((unused))
#endif

#if !defined(BUNUSED_TYPE_ALIAS)
#define BUNUSED_TYPE_ALIAS
#endif

/* BFUNCTION_SIGNATURE */

#if !defined(BFUNCTION_SIGNATURE)
#if BCOMPILER(GCC_COMPATIBLE)
#define BFUNCTION_SIGNATURE __PRETTY_FUNCTION__
#elif BCOMPILER(MSVC)
#define BFUNCTION_SIGNATURE __FUNCSIG__
#else
#error "Unsupported compiler"
#endif
#endif

/* BALLOW_DEPRECATED_DECLARATIONS_BEGIN and BALLOW_DEPRECATED_DECLARATIONS_END */

#if BCOMPILER(GCC_COMPATIBLE)
#define BALLOW_DEPRECATED_DECLARATIONS_BEGIN \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#define BALLOW_DEPRECATED_DECLARATIONS_END \
    _Pragma("GCC diagnostic pop")
#else
#define BALLOW_DEPRECATED_DECLARATIONS_BEGIN
#define BALLOW_DEPRECATED_DECLARATIONS_END
#endif

/* BALLOW_UNSAFE_BUFFER_USAGE_BEGIN */

#if BCOMPILER(CLANG)
#define BALLOW_UNSAFE_BUFFER_USAGE_BEGIN \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Wunsafe-buffer-usage\"")

#define BALLOW_UNSAFE_BUFFER_USAGE_END \
    _Pragma("clang diagnostic pop")
#else
#define BALLOW_UNSAFE_BUFFER_USAGE_BEGIN
#define BALLOW_UNSAFE_BUFFER_USAGE_END
#endif

/* MUST_TAIL_CALL */

// 32-bit platforms use different calling conventions, so a MUST_TAIL_CALL function
// written for 64-bit may fail to tail call on 32-bit.
// It also doesn't work on ppc64le: https://github.com/llvm/llvm-project/issues/98859
// and on Windows: https://github.com/llvm/llvm-project/issues/116568
#if BCOMPILER(CLANG)
#if __SIZEOF_POINTER__ == 8
#if !defined(BMUST_TAIL_CALL) && defined(__cplusplus) && defined(__has_cpp_attribute)
#if __has_cpp_attribute(clang::musttail) && !defined(__powerpc__) && !defined(_WIN32)
#define BMUST_TAIL_CALL [[clang::musttail]]
#endif
#endif
#endif
#endif

#if !defined(BMUST_TAIL_CALL)
#define BMUST_TAIL_CALL
#endif

