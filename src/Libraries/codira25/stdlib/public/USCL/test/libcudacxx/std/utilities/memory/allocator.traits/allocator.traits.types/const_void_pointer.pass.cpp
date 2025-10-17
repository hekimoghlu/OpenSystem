/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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
//     typedef Alloc::const_void_pointer
//           | pointer_traits<pointer>::rebind<const void>
//                                          const_void_pointer;
//     ...
// };

#include <uscl/std/__memory_>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
struct Ptr
{};

template <class T>
struct A
{
  typedef T value_type;
  typedef Ptr<T> pointer;
};

template <class T>
struct B
{
  typedef T value_type;
};

template <class T>
struct CPtr
{};

template <class T>
struct C
{
  typedef T value_type;
  typedef CPtr<const void> const_void_pointer;
};

template <class T>
struct D
{
  typedef T value_type;

private:
  typedef int const_void_pointer;
};

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<A<char>>::const_void_pointer, Ptr<const void>>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<B<char>>::const_void_pointer, const void*>::value), "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<C<char>>::const_void_pointer, CPtr<const void>>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<D<char>>::const_void_pointer, const void*>::value), "");
  return 0;
}
