/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 1, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class T>
// concept destructible = is_nothrow_destructible_v<T>;

#include <uscl/std/concepts>
#include <uscl/std/type_traits>

struct Empty
{};

struct Defaulted
{
  ~Defaulted() = default;
};
struct Deleted
{
  ~Deleted() = delete;
};

struct Noexcept
{
  __host__ __device__ ~Noexcept() noexcept;
};
struct NoexceptTrue
{
  __host__ __device__ ~NoexceptTrue() noexcept(true);
};
struct NoexceptFalse
{
  __host__ __device__ ~NoexceptFalse() noexcept(false);
};

struct Protected
{
protected:
  ~Protected() = default;
};
struct Private
{
private:
  ~Private() = default;
};

template <class T>
struct NoexceptDependant
{
  __host__ __device__ ~NoexceptDependant() noexcept(cuda::std::is_same_v<T, int>);
};

template <class T>
__host__ __device__ void test()
{
  static_assert(cuda::std::destructible<T> == cuda::std::is_nothrow_destructible_v<T>, "");
}

__host__ __device__ void test()
{
  test<Empty>();

  test<Defaulted>();
  test<Deleted>();

  test<Noexcept>();
  test<NoexceptTrue>();
  test<NoexceptFalse>();

  test<Protected>();
  test<Private>();

  test<NoexceptDependant<int>>();
  test<NoexceptDependant<double>>();

  test<bool>();
  test<char>();
  test<int>();
  test<double>();
}

// Required for MSVC internal test runner compatibility.
int main(int, char**)
{
  return 0;
}
