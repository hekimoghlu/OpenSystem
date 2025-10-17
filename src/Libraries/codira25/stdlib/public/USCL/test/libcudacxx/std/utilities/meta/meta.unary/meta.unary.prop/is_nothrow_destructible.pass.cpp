/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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

// type_traits

// is_nothrow_destructible

#include <uscl/std/type_traits>

#include "test_macros.h"

// Prevent warning when testing the Abstract test type.
TEST_DIAG_SUPPRESS_CLANG("-Wdelete-non-virtual-dtor")

template <class T>
__host__ __device__ void test_is_nothrow_destructible()
{
  static_assert(cuda::std::is_nothrow_destructible<T>::value, "");
  static_assert(cuda::std::is_nothrow_destructible<const T>::value, "");
  static_assert(cuda::std::is_nothrow_destructible<volatile T>::value, "");
  static_assert(cuda::std::is_nothrow_destructible<const volatile T>::value, "");
  static_assert(cuda::std::is_nothrow_destructible_v<T>, "");
  static_assert(cuda::std::is_nothrow_destructible_v<const T>, "");
  static_assert(cuda::std::is_nothrow_destructible_v<volatile T>, "");
  static_assert(cuda::std::is_nothrow_destructible_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_nothrow_destructible()
{
  static_assert(!cuda::std::is_nothrow_destructible<T>::value, "");
  static_assert(!cuda::std::is_nothrow_destructible<const T>::value, "");
  static_assert(!cuda::std::is_nothrow_destructible<volatile T>::value, "");
  static_assert(!cuda::std::is_nothrow_destructible<const volatile T>::value, "");
  static_assert(!cuda::std::is_nothrow_destructible_v<T>, "");
  static_assert(!cuda::std::is_nothrow_destructible_v<const T>, "");
  static_assert(!cuda::std::is_nothrow_destructible_v<volatile T>, "");
  static_assert(!cuda::std::is_nothrow_destructible_v<const volatile T>, "");
}

struct PublicDestructor
{
public:
  __host__ __device__ ~PublicDestructor() {}
};
struct ProtectedDestructor
{
protected:
  __host__ __device__ ~ProtectedDestructor() {}
};
struct PrivateDestructor
{
private:
  __host__ __device__ ~PrivateDestructor() {}
};

struct VirtualPublicDestructor
{
public:
  __host__ __device__ virtual ~VirtualPublicDestructor() {}
};
struct VirtualProtectedDestructor
{
protected:
  __host__ __device__ virtual ~VirtualProtectedDestructor() {}
};
struct VirtualPrivateDestructor
{
private:
  __host__ __device__ virtual ~VirtualPrivateDestructor() {}
};

struct PurePublicDestructor
{
public:
  __host__ __device__ virtual ~PurePublicDestructor() = 0;
};
struct PureProtectedDestructor
{
protected:
  __host__ __device__ virtual ~PureProtectedDestructor() = 0;
};
struct PurePrivateDestructor
{
private:
  __host__ __device__ virtual ~PurePrivateDestructor() = 0;
};

class Empty
{};

union Union
{};

struct bit_zero
{
  int : 0;
};

class Abstract
{
  __host__ __device__ virtual void foo() = 0;
};

int main(int, char**)
{
  test_is_not_nothrow_destructible<void>();
  test_is_not_nothrow_destructible<char[]>();
  test_is_not_nothrow_destructible<char[][3]>();

  test_is_nothrow_destructible<int&>();
  test_is_nothrow_destructible<int>();
  test_is_nothrow_destructible<double>();
  test_is_nothrow_destructible<int*>();
  test_is_nothrow_destructible<const int*>();
  test_is_nothrow_destructible<char[3]>();

  // requires noexcept. These are all destructible.
  test_is_nothrow_destructible<PublicDestructor>();
  test_is_nothrow_destructible<VirtualPublicDestructor>();
  test_is_nothrow_destructible<PurePublicDestructor>();
  test_is_nothrow_destructible<bit_zero>();
  test_is_nothrow_destructible<Abstract>();
  test_is_nothrow_destructible<Empty>();
  test_is_nothrow_destructible<Union>();

  // requires access control
  test_is_not_nothrow_destructible<ProtectedDestructor>();
  test_is_not_nothrow_destructible<PrivateDestructor>();
  test_is_not_nothrow_destructible<VirtualProtectedDestructor>();
  test_is_not_nothrow_destructible<VirtualPrivateDestructor>();
  test_is_not_nothrow_destructible<PureProtectedDestructor>();
  test_is_not_nothrow_destructible<PurePrivateDestructor>();

  return 0;
}
