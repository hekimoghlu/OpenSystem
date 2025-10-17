/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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

// REQUIRES: c++03 || c++11 || c++14 || c++17

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

// <functional>

// reference_wrapper

// check that binder typedefs exit

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <uscl/std/functional>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

struct UnaryFunction
{
  typedef long argument_type;
  typedef char result_type;
};

struct BinaryFunction
{
  typedef int first_argument_type;
  typedef char second_argument_type;
  typedef long result_type;
};

static_assert(cuda::std::is_same<cuda::std::reference_wrapper<int (UnaryFunction::*)()>::result_type, int>::value, "");
static_assert(
  cuda::std::is_same<cuda::std::reference_wrapper<int (UnaryFunction::*)()>::argument_type, UnaryFunction*>::value, "");

static_assert(cuda::std::is_same<cuda::std::reference_wrapper<int (BinaryFunction::*)(char)>::result_type, int>::value,
              "");
static_assert(cuda::std::is_same<cuda::std::reference_wrapper<int (BinaryFunction::*)(char)>::first_argument_type,
                                 BinaryFunction*>::value,
              "");
static_assert(
  cuda::std::is_same<cuda::std::reference_wrapper<int (BinaryFunction::*)(char)>::second_argument_type, char>::value,
  "");

static_assert(cuda::std::is_same<cuda::std::reference_wrapper<void (*)()>::result_type, void>::value, "");

int main(int, char**)
{
  return 0;
}
