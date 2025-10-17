/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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

// <cuda/std/tuple>

// template <class... Types> class tuple;

// explicit tuple(const T&...);

#include <uscl/std/cassert>
#include <uscl/std/tuple>

struct ExplicitCopy
{
  __host__ __device__ ExplicitCopy(int) {}
  __host__ __device__ explicit ExplicitCopy(ExplicitCopy const&) {}
};

__host__ __device__ std::tuple<ExplicitCopy> const_explicit_copy()
{
  const ExplicitCopy e(42);
  return {e};
  // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}

__host__ __device__ std::tuple<ExplicitCopy> non_const_explicit_copy()
{
  ExplicitCopy e(42);
  return {e};
  // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}

__host__ __device__ std::tuple<ExplicitCopy> const_explicit_copy_no_brace()
{
  const ExplicitCopy e(42);
  return e;
  // expected-error@-1 {{no viable conversion}}
}

int main(int, char**)
{
  return 0;
}
