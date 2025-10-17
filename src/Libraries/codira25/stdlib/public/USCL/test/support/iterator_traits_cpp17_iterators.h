/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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

#ifndef TEST_SUPPORT_ITERATOR_TRAITS_ITERATOR_TRAITS_CPP17_ITERATORS
#define TEST_SUPPORT_ITERATOR_TRAITS_ITERATOR_TRAITS_CPP17_ITERATORS

struct iterator_traits_cpp17_iterator
{
  __host__ __device__ int& operator*();
  __host__ __device__ iterator_traits_cpp17_iterator& operator++();
  __host__ __device__ iterator_traits_cpp17_iterator operator++(int);
};

struct iterator_traits_cpp17_proxy_iterator
{
  __host__ __device__ int operator*();
  __host__ __device__ iterator_traits_cpp17_proxy_iterator& operator++();

  // this returns legcay_iterator, not iterator_traits_cpp17_proxy_iterator
  __host__ __device__ iterator_traits_cpp17_iterator operator++(int);
};

struct iterator_traits_cpp17_input_iterator
{
  using difference_type = int;
  using value_type      = long;

  __host__ __device__ int& operator*();
  __host__ __device__ iterator_traits_cpp17_input_iterator& operator++();
  __host__ __device__ iterator_traits_cpp17_input_iterator operator++(int);

  __host__ __device__ bool operator==(iterator_traits_cpp17_input_iterator const&) const;
#if TEST_STD_VER < 2020
  __host__ __device__ bool operator!=(iterator_traits_cpp17_input_iterator const&) const;
#endif
};

struct iterator_traits_cpp17_proxy_input_iterator
{
  using difference_type = int;
  using value_type      = long;

  __host__ __device__ int operator*();
  __host__ __device__ iterator_traits_cpp17_proxy_input_iterator& operator++();

  // this returns legcay_input_iterator, not iterator_traits_cpp17_proxy_input_iterator
  __host__ __device__ iterator_traits_cpp17_input_iterator operator++(int);

  __host__ __device__ bool operator==(iterator_traits_cpp17_proxy_input_iterator const&) const;
#if TEST_STD_VER < 2020
  __host__ __device__ bool operator!=(iterator_traits_cpp17_proxy_input_iterator const&) const;
#endif
};

struct iterator_traits_cpp17_forward_iterator
{
  using difference_type = int;
  using value_type      = int;

  __host__ __device__ int& operator*();
  __host__ __device__ iterator_traits_cpp17_forward_iterator& operator++();
  __host__ __device__ iterator_traits_cpp17_forward_iterator operator++(int);

  __host__ __device__ bool operator==(iterator_traits_cpp17_forward_iterator const&) const;
#if TEST_STD_VER < 2020
  __host__ __device__ bool operator!=(iterator_traits_cpp17_forward_iterator const&) const;
#endif
};

struct iterator_traits_cpp17_bidirectional_iterator
{
  using difference_type = int;
  using value_type      = int;

  __host__ __device__ int& operator*();
  __host__ __device__ iterator_traits_cpp17_bidirectional_iterator& operator++();
  __host__ __device__ iterator_traits_cpp17_bidirectional_iterator operator++(int);
  __host__ __device__ iterator_traits_cpp17_bidirectional_iterator& operator--();
  __host__ __device__ iterator_traits_cpp17_bidirectional_iterator operator--(int);

  __host__ __device__ bool operator==(iterator_traits_cpp17_bidirectional_iterator const&) const;
#if TEST_STD_VER < 2020
  __host__ __device__ bool operator!=(iterator_traits_cpp17_bidirectional_iterator const&) const;
#endif
};

struct iterator_traits_cpp17_random_access_iterator
{
  using difference_type = int;
  using value_type      = int;

  __host__ __device__ int& operator*();
  __host__ __device__ int& operator[](difference_type);
  __host__ __device__ iterator_traits_cpp17_random_access_iterator& operator++();
  __host__ __device__ iterator_traits_cpp17_random_access_iterator operator++(int);
  __host__ __device__ iterator_traits_cpp17_random_access_iterator& operator--();
  __host__ __device__ iterator_traits_cpp17_random_access_iterator operator--(int);

  __host__ __device__ bool operator==(iterator_traits_cpp17_random_access_iterator const&) const;
#if TEST_STD_VER < 2020
  __host__ __device__ bool operator!=(iterator_traits_cpp17_random_access_iterator const&) const;
#endif
  __host__ __device__ bool operator<(iterator_traits_cpp17_random_access_iterator const&) const;
  __host__ __device__ bool operator>(iterator_traits_cpp17_random_access_iterator const&) const;
  __host__ __device__ bool operator<=(iterator_traits_cpp17_random_access_iterator const&) const;
  __host__ __device__ bool operator>=(iterator_traits_cpp17_random_access_iterator const&) const;

  __host__ __device__ iterator_traits_cpp17_random_access_iterator& operator+=(difference_type);
  __host__ __device__ iterator_traits_cpp17_random_access_iterator& operator-=(difference_type);

  __host__ __device__ friend iterator_traits_cpp17_random_access_iterator
  operator+(iterator_traits_cpp17_random_access_iterator, difference_type);
  __host__ __device__ friend iterator_traits_cpp17_random_access_iterator
  operator+(difference_type, iterator_traits_cpp17_random_access_iterator);
  __host__ __device__ friend iterator_traits_cpp17_random_access_iterator
  operator-(iterator_traits_cpp17_random_access_iterator, difference_type);
  __host__ __device__ friend difference_type
  operator-(iterator_traits_cpp17_random_access_iterator, iterator_traits_cpp17_random_access_iterator);
};

#endif // TEST_SUPPORT_ITERATOR_TRAITS_ITERATOR_TRAITS_CPP17_ITERATORS
