/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 27, 2023.
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

// template<class F, class... Args>
// concept equivalence_relation;

#include <uscl/std/concepts>

using cuda::std::equivalence_relation;

static_assert(equivalence_relation<bool(int, int), int, int>, "");
static_assert(equivalence_relation<bool(int, int), double, double>, "");
static_assert(equivalence_relation<bool(int, double), double, double>, "");

static_assert(!equivalence_relation<bool (*)(), int, double>, "");
static_assert(!equivalence_relation<bool (*)(int), int, double>, "");
static_assert(!equivalence_relation<bool (*)(double), int, double>, "");

static_assert(!equivalence_relation<bool(double, double*), double, double*>, "");
static_assert(!equivalence_relation<bool(int&, int&), double&, double&>, "");

struct S1
{};
static_assert(cuda::std::relation<bool (S1::*)(S1*), S1*, S1*>, "");
static_assert(cuda::std::relation<bool (S1::*)(S1&), S1&, S1&>, "");

struct S2
{};

struct P1
{
  __host__ __device__ bool operator()(S1, S1) const;
};
static_assert(equivalence_relation<P1, S1, S1>, "");

struct P2
{
  __host__ __device__ bool operator()(S1, S1) const;
  __host__ __device__ bool operator()(S1, S2) const;
};
static_assert(!equivalence_relation<P2, S1, S2>, "");

struct P3
{
  __host__ __device__ bool operator()(S1, S1) const;
  __host__ __device__ bool operator()(S1, S2) const;
  __host__ __device__ bool operator()(S2, S1) const;
};
static_assert(!equivalence_relation<P3, S1, S2>, "");

struct P4
{
  __host__ __device__ bool operator()(S1, S1) const;
  __host__ __device__ bool operator()(S1, S2) const;
  __host__ __device__ bool operator()(S2, S1) const;
  __host__ __device__ bool operator()(S2, S2) const;
};
static_assert(equivalence_relation<P4, S1, S2>, "");

int main(int, char**)
{
  return 0;
}
