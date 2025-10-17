/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 23, 2025.
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

// libstdc++ uses the non-constexpr function std::__glibcxx_assert_fail()
// to trigger compilation errors when the __glibcxx_assert(cond) macro
// is used in a constexpr context.
// Compilation fails when using code from the libstdc++ (such as std::array) on
// device code, since these assertions invoke a non-constexpr host function from
// device code.
//
// To work around this issue, we declare our own device version of the function

#ifndef __CLANG_CUDA_WRAPPERS_BITS_CPP_CONFIG
#define __CLANG_CUDA_WRAPPERS_BITS_CPP_CONFIG

#include_next <bits/c++config.h>

#ifdef _LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_NAMESPACE_STD
#else
namespace std {
#ifdef _GLIBCXX_BEGIN_NAMESPACE_VERSION
_GLIBCXX_BEGIN_NAMESPACE_VERSION
#endif

#pragma push_macro("CUDA_NOEXCEPT")
#if __cplusplus >= 201103L
#define CUDA_NOEXCEPT noexcept
#else
#define CUDA_NOEXCEPT
#endif

__attribute__((device, noreturn)) inline void
__glibcxx_assert_fail(const char *file, int line, const char *function,
                      const char *condition) CUDA_NOEXCEPT {
#ifdef _GLIBCXX_VERBOSE_ASSERT
  if (file && function && condition)
    __builtin_printf("%s:%d: %s: Assertion '%s' failed.\n", file, line,
                     function, condition);
  else if (function)
    __builtin_printf("%s: Undefined behavior detected.\n", function);
#endif
  __builtin_abort();
}

#endif
__attribute__((device, noreturn, __always_inline__,
               __visibility__("default"))) inline void
__glibcxx_assert_fail() CUDA_NOEXCEPT {
  __builtin_abort();
}

#pragma pop_macro("CUDA_NOEXCEPT")

#ifdef _LIBCPP_END_NAMESPACE_STD
_LIBCPP_END_NAMESPACE_STD
#else
#ifdef _GLIBCXX_BEGIN_NAMESPACE_VERSION
_GLIBCXX_END_NAMESPACE_VERSION
#endif
} // namespace std
#endif

#endif
