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
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// ~optional();

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

struct PODType
{
  int value;
  int value2;
};

class X
{
public:
  STATIC_MEMBER_VAR(dtor_called, bool)
  X() = default;
  __host__ __device__ ~X()
  {
    dtor_called() = true;
  }
};

int main(int, char**)
{
  {
    typedef int T;
    static_assert(cuda::std::is_trivially_destructible<T>::value, "");
    static_assert(cuda::std::is_trivially_destructible<optional<T>>::value, "");
#ifdef CCCL_ENABLE_OPTIONAL_REF
    static_assert(cuda::std::is_trivially_destructible<optional<T&>>::value, "");
#endif // CCCL_ENABLE_OPTIONAL_REF
  }
  {
    typedef double T;
    static_assert(cuda::std::is_trivially_destructible<T>::value, "");
    static_assert(cuda::std::is_trivially_destructible<optional<T>>::value, "");
#ifdef CCCL_ENABLE_OPTIONAL_REF
    static_assert(cuda::std::is_trivially_destructible<optional<T&>>::value, "");
#endif // CCCL_ENABLE_OPTIONAL_REF
  }
  {
    typedef PODType T;
    static_assert(cuda::std::is_trivially_destructible<T>::value, "");
    static_assert(cuda::std::is_trivially_destructible<optional<T>>::value, "");
#ifdef CCCL_ENABLE_OPTIONAL_REF
    static_assert(cuda::std::is_trivially_destructible<optional<T&>>::value, "");
#endif // CCCL_ENABLE_OPTIONAL_REF
  }
  {
    typedef X T;
    static_assert(!cuda::std::is_trivially_destructible<T>::value, "");
    static_assert(!cuda::std::is_trivially_destructible<optional<T>>::value, "");
#ifdef CCCL_ENABLE_OPTIONAL_REF
    static_assert(cuda::std::is_trivially_destructible<optional<T&>>::value, "");
#endif // CCCL_ENABLE_OPTIONAL_REF
    {
      X x;
      optional<X> opt{x};
      assert(X::dtor_called() == false);
    }
    assert(X::dtor_called() == true);
  }

  return 0;
}
