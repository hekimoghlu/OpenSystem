/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <class T, size_t N >
// struct array
// {
//     // types:
//     typedef T& reference;
//     typedef const T& const_reference;
//     typedef implementation defined iterator;
//     typedef implementation defined const_iterator;
//     typedef T value_type;
//     typedef T* pointer;
//     typedef size_t size_type;
//     typedef ptrdiff_t difference_type;
//     typedef T value_type;
//     typedef cuda::std::reverse_iterator<iterator> reverse_iterator;
//     typedef cuda::std::reverse_iterator<const_iterator> const_reverse_iterator;

#include <uscl/std/array>
#include <uscl/std/iterator>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class C>
__host__ __device__ void test_iterators()
{
  typedef cuda::std::iterator_traits<typename C::iterator> ItT;
  typedef cuda::std::iterator_traits<typename C::const_iterator> CItT;
  static_assert((cuda::std::is_same<typename ItT::iterator_category, cuda::std::random_access_iterator_tag>::value),
                "");
  static_assert((cuda::std::is_same<typename ItT::value_type, typename C::value_type>::value), "");
  static_assert((cuda::std::is_same<typename ItT::reference, typename C::reference>::value), "");
  static_assert((cuda::std::is_same<typename ItT::pointer, typename C::pointer>::value), "");
  static_assert((cuda::std::is_same<typename ItT::difference_type, typename C::difference_type>::value), "");

  static_assert((cuda::std::is_same<typename CItT::iterator_category, cuda::std::random_access_iterator_tag>::value),
                "");
  static_assert((cuda::std::is_same<typename CItT::value_type, typename C::value_type>::value), "");
  static_assert((cuda::std::is_same<typename CItT::reference, typename C::const_reference>::value), "");
  static_assert((cuda::std::is_same<typename CItT::pointer, typename C::const_pointer>::value), "");
  static_assert((cuda::std::is_same<typename CItT::difference_type, typename C::difference_type>::value), "");
}

int main(int, char**)
{
  {
    typedef double T;
    typedef cuda::std::array<T, 10> C;
    static_assert((cuda::std::is_same<C::reference, T&>::value), "");
    static_assert((cuda::std::is_same<C::const_reference, const T&>::value), "");
    static_assert((cuda::std::is_same<C::iterator, T*>::value), "");
    static_assert((cuda::std::is_same<C::const_iterator, const T*>::value), "");
    test_iterators<C>();
    static_assert((cuda::std::is_same<C::pointer, T*>::value), "");
    static_assert((cuda::std::is_same<C::const_pointer, const T*>::value), "");
    static_assert((cuda::std::is_same<C::size_type, cuda::std::size_t>::value), "");
    static_assert((cuda::std::is_same<C::difference_type, cuda::std::ptrdiff_t>::value), "");
    static_assert((cuda::std::is_same<C::reverse_iterator, cuda::std::reverse_iterator<C::iterator>>::value), "");
    static_assert(
      (cuda::std::is_same<C::const_reverse_iterator, cuda::std::reverse_iterator<C::const_iterator>>::value), "");

    static_assert((cuda::std::is_signed<typename C::difference_type>::value), "");
    static_assert((cuda::std::is_unsigned<typename C::size_type>::value), "");
    static_assert(
      (cuda::std::is_same<typename C::difference_type,
                          typename cuda::std::iterator_traits<typename C::iterator>::difference_type>::value),
      "");
    static_assert(
      (cuda::std::is_same<typename C::difference_type,
                          typename cuda::std::iterator_traits<typename C::const_iterator>::difference_type>::value),
      "");
  }
  {
    typedef int* T;
    typedef cuda::std::array<T, 0> C;
    static_assert((cuda::std::is_same<C::reference, T&>::value), "");
    static_assert((cuda::std::is_same<C::const_reference, const T&>::value), "");
    static_assert((cuda::std::is_same<C::iterator, T*>::value), "");
    static_assert((cuda::std::is_same<C::const_iterator, const T*>::value), "");
    test_iterators<C>();
    static_assert((cuda::std::is_same<C::pointer, T*>::value), "");
    static_assert((cuda::std::is_same<C::const_pointer, const T*>::value), "");
    static_assert((cuda::std::is_same<C::size_type, cuda::std::size_t>::value), "");
    static_assert((cuda::std::is_same<C::difference_type, cuda::std::ptrdiff_t>::value), "");
    static_assert((cuda::std::is_same<C::reverse_iterator, cuda::std::reverse_iterator<C::iterator>>::value), "");
    static_assert(
      (cuda::std::is_same<C::const_reverse_iterator, cuda::std::reverse_iterator<C::const_iterator>>::value), "");

    static_assert((cuda::std::is_signed<typename C::difference_type>::value), "");
    static_assert((cuda::std::is_unsigned<typename C::size_type>::value), "");
    static_assert(
      (cuda::std::is_same<typename C::difference_type,
                          typename cuda::std::iterator_traits<typename C::iterator>::difference_type>::value),
      "");
    static_assert(
      (cuda::std::is_same<typename C::difference_type,
                          typename cuda::std::iterator_traits<typename C::const_iterator>::difference_type>::value),
      "");
  }

  return 0;
}
