/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: gcc-6
// UNSUPPORTED: windows
// We run out of heap space with windows

// <cuda/std/tuple>

// template <class F, class T> constexpr decltype(auto) apply(F &&, T &&)

// Stress testing large arities with tuple and array.

#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/tuple>
#include <uscl/std/utility>

#include "test_macros.h"

////////////////////////////////////////////////////////////////////////////////
template <class T, cuda::std::size_t Dummy = 0>
struct always_imp
{
  using type = T;
};

template <class T, cuda::std::size_t Dummy = 0>
using always_t = typename always_imp<T, Dummy>::type;

////////////////////////////////////////////////////////////////////////////////
template <class Tuple, class Idx>
struct make_function;

template <class Tp, cuda::std::size_t... Idx>
struct make_function<Tp, cuda::std::integer_sequence<cuda::std::size_t, Idx...>>
{
  using type = bool (*)(always_t<Tp, Idx>...);
};

template <class Tp, cuda::std::size_t Size>
using make_function_t = typename make_function<Tp, cuda::std::make_index_sequence<Size>>::type;

////////////////////////////////////////////////////////////////////////////////
template <class Tp, class Idx>
struct make_tuple_imp;

////////////////////////////////////////////////////////////////////////////////
template <class Tp, cuda::std::size_t... Idx>
struct make_tuple_imp<Tp, cuda::std::integer_sequence<cuda::std::size_t, Idx...>>
{
  using type = cuda::std::tuple<always_t<Tp, Idx>...>;
};

template <class Tp, cuda::std::size_t Size>
using make_tuple_t = typename make_tuple_imp<Tp, cuda::std::make_index_sequence<Size>>::type;

template <class... Types>
__host__ __device__ bool test_apply_fn(Types...)
{
  return true;
}

template <cuda::std::size_t Size>
__host__ __device__ void test_all()
{
  using A      = cuda::std::array<int, Size>;
  using ConstA = cuda::std::array<int const, Size>;

  using Tuple  = make_tuple_t<int, Size>;
  using CTuple = make_tuple_t<const int, Size>;

  using ValFn  = make_function_t<int, Size>;
  ValFn val_fn = &test_apply_fn;

  using RefFn  = make_function_t<int&, Size>;
  RefFn ref_fn = &test_apply_fn;

  using CRefFn   = make_function_t<int const&, Size>;
  CRefFn cref_fn = &test_apply_fn;

  using RRefFn   = make_function_t<int&&, Size>;
  RRefFn rref_fn = &test_apply_fn;

  {
    A a{};
    assert(cuda::std::apply(val_fn, a));
    assert(cuda::std::apply(ref_fn, a));
    assert(cuda::std::apply(cref_fn, a));
    assert(cuda::std::apply(rref_fn, cuda::std::move(a)));
  }
  {
    ConstA a{};
    assert(cuda::std::apply(val_fn, a));
    assert(cuda::std::apply(cref_fn, a));
  }
  {
    Tuple a{};
    assert(cuda::std::apply(val_fn, a));
    assert(cuda::std::apply(ref_fn, a));
    assert(cuda::std::apply(cref_fn, a));
    assert(cuda::std::apply(rref_fn, cuda::std::move(a)));
  }
  {
    CTuple a{};
    assert(cuda::std::apply(val_fn, a));
    assert(cuda::std::apply(cref_fn, a));
  }
}

template <cuda::std::size_t Size>
__host__ __device__ void test_one()
{
  using A     = cuda::std::array<int, Size>;
  using Tuple = make_tuple_t<int, Size>;

  using ValFn  = make_function_t<int, Size>;
  ValFn val_fn = &test_apply_fn;

  {
    A a{};
    assert(cuda::std::apply(val_fn, a));
  }
  {
    Tuple a{};
    assert(cuda::std::apply(val_fn, a));
  }
}

int main(int, char**)
{
  // Instantiate with 1-5 arguments.
  test_all<1>();
  test_all<2>();
  test_all<3>();
  test_all<4>();
  test_all<5>();

  // Stress test with 256
  test_one<256>();

  return 0;
}
