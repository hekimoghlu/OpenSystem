/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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

// Before GCC 6, aggregate initialization kicks in.
// See https://stackoverflow.com/q/41799015/627587.
// UNSUPPORTED: gcc-5

// <cuda/std/tuple>

// template <class... Types> class tuple;

// explicit(see-below) constexpr tuple();

#include <uscl/std/tuple>

struct Implicit
{
  Implicit() = default;
};

struct Explicit
{
  explicit Explicit() = default;
};

__host__ __device__ cuda::std::tuple<> test1()
{
  return {};
}

__host__ __device__ cuda::std::tuple<Implicit> test2()
{
  return {};
}
__host__ __device__ cuda::std::tuple<Explicit> test3()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}

__host__ __device__ cuda::std::tuple<Implicit, Implicit> test4()
{
  return {};
}
__host__ __device__ cuda::std::tuple<Explicit, Implicit> test5()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Implicit, Explicit> test6()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Explicit, Explicit> test7()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}

__host__ __device__ cuda::std::tuple<Implicit, Implicit, Implicit> test8()
{
  return {};
}
__host__ __device__ cuda::std::tuple<Implicit, Implicit, Explicit> test9()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Implicit, Explicit, Implicit> test10()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Implicit, Explicit, Explicit> test11()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Explicit, Implicit, Implicit> test12()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Explicit, Implicit, Explicit> test13()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Explicit, Explicit, Implicit> test14()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Explicit, Explicit, Explicit> test15()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}

int main(int, char**)
{
  return 0;
}
