/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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
// <memory>

// template <class Alloc>
// struct allocator_traits
// {
//     template <class T> using rebind_traits = allocator_traits<rebind_alloc<T>>;
//     ...
// };

#include <uscl/std/__memory_>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
struct ReboundA
{};

template <class T>
struct A
{
  typedef T value_type;

  template <class U>
  struct rebind
  {
    typedef ReboundA<U> other;
  };
};

template <class T, class U>
struct ReboundB
{};

template <class T, class U>
struct B
{
  typedef T value_type;

  template <class V>
  struct rebind
  {
    typedef ReboundB<V, U> other;
  };
};

template <class T>
struct C
{
  typedef T value_type;
};

template <class T, class U>
struct D
{
  typedef T value_type;
};

template <class T>
struct E
{
  typedef T value_type;

  template <class U>
  struct rebind
  {
    typedef ReboundA<U> otter;
  };
};

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<A<char>>::rebind_traits<double>,
                                    cuda::std::allocator_traits<ReboundA<double>>>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<B<int, char>>::rebind_traits<double>,
                                    cuda::std::allocator_traits<ReboundB<double, char>>>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<C<char>>::rebind_traits<double>,
                                    cuda::std::allocator_traits<C<double>>>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<D<int, char>>::rebind_traits<double>,
                                    cuda::std::allocator_traits<D<double, char>>>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<E<char>>::rebind_traits<double>,
                                    cuda::std::allocator_traits<E<double>>>::value),
                "");

  return 0;
}
