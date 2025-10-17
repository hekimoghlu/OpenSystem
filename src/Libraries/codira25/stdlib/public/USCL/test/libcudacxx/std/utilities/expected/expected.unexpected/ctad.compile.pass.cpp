/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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

// UNSUPPORTED: gcc-6, gcc-7, gcc-8, gcc-9

// template<class E> unexpected(E) -> unexpected<E>;

#include <uscl/std/concepts>
#include <uscl/std/expected>

struct Foo
{};

static_assert(cuda::std::same_as<decltype(cuda::std::unexpected(5)), cuda::std::unexpected<int>>);
static_assert(cuda::std::same_as<decltype(cuda::std::unexpected(Foo{})), cuda::std::unexpected<Foo>>);
static_assert(
  cuda::std::same_as<decltype(cuda::std::unexpected(cuda::std::unexpected<int>(5))), cuda::std::unexpected<int>>);

int main(int, char**)
{
  return 0;
}
