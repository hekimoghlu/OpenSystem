/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// basic_format_arg() noexcept;

// The class has several exposition only private constructors. These are tested
// in visit_format_arg.pass.cpp

#include <uscl/std/__format_>
#include <uscl/std/cassert>
#include <uscl/std/type_traits>

template <class CharT>
__host__ __device__ void test_constructor()
{
  using Context = cuda::std::basic_format_context<CharT*, CharT>;

  static_assert(cuda::std::is_nothrow_default_constructible_v<cuda::std::basic_format_arg<Context>>);

  cuda::std::basic_format_arg<Context> format_arg{};
  assert(!format_arg);
}

__host__ __device__ void test()
{
  test_constructor<char>();
#if _CCCL_HAS_CHAR8_T()
  test_constructor<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_constructor<char16_t>();
  test_constructor<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test_constructor<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

int main(int, char**)
{
  test();
  return 0;
}
