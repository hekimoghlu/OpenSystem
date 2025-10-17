/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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

// UNSUPPORTED: nvrtc

// <cuda/std/format>

// class format_error;

#include <uscl/std/__format_>
#include <uscl/std/cassert>
#include <uscl/std/cstring>
#include <uscl/std/type_traits>

#include <string>

#include "test_macros.h"

void test_format_error()
{
#if __cpp_lib_format >= 201907L
  static_assert(cuda::std::is_same_v<cuda::std::format_error, std::format_error>);
#endif // __cpp_lib_format >= 201907L

  static_assert(cuda::std::is_base_of_v<std::runtime_error, cuda::std::format_error>);
  static_assert(cuda::std::is_polymorphic_v<cuda::std::format_error>);

  {
    const char* msg = "format_error message c-string";
    cuda::std::format_error e(msg);
    assert(cuda::std::strcmp(e.what(), msg) == 0);
    cuda::std::format_error e2(e);
    assert(cuda::std::strcmp(e2.what(), msg) == 0);
    e2 = e;
    assert(cuda::std::strcmp(e2.what(), msg) == 0);
  }
  {
    std::string msg("format_error message std::string");
    cuda::std::format_error e(msg);
    assert(e.what() == msg);
    cuda::std::format_error e2(e);
    assert(e2.what() == msg);
    e2 = e;
    assert(e2.what() == msg);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_format_error();))
  return 0;
}
