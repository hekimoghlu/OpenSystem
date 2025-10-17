/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// UNSUPPORTED: nvrtc

// const char* what() const noexcept override;

#include <uscl/std/expected>
#include <uscl/std/utility>

#include "test_macros.h"

#if TEST_HAS_EXCEPTIONS()
template <class T, class = void>
constexpr bool WhatNoexcept = false;

template <class T>
constexpr bool WhatNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<const T&>().what())>> =
  noexcept(cuda::std::declval<const T&>().what());

struct foo
{};

static_assert(!WhatNoexcept<foo>, "");
static_assert(WhatNoexcept<cuda::std::bad_expected_access<int>>, "");
static_assert(WhatNoexcept<cuda::std::bad_expected_access<foo>>, "");
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  return 0;
}
