/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include <uscl/std/array>

#include "MoveOnly.h"
#include "test_macros.h"

// expected-warning@array:* 0-1 {{suggest braces around initialization of subobject}}

int main(int, char**)
{
  {
    char source[3][6] = {"hi", "world"};
    // expected-error@array:* {{to_array does not accept multidimensional arrays}}
    // expected-error@array:* {{to_array requires copy constructible elements}}
    // expected-error@array:* 3 {{cannot initialize}}
    cuda::std::to_array(source); // expected-note {{requested here}}
  }

  {
    MoveOnly mo[] = {MoveOnly{3}};
    // expected-error@array:* {{to_array requires copy constructible elements}}
    // expected-error-re@array:* {{{{(call to implicitly-deleted copy constructor of 'MoveOnly')|(call to deleted
    // constructor of 'MoveOnly')}}}}
    cuda::std::to_array(mo); // expected-note {{requested here}}
  }

  {
    const MoveOnly cmo[] = {MoveOnly{3}};
    // expected-error@array:* {{to_array requires move constructible elements}}
    // expected-error-re@array:* {{{{(call to implicitly-deleted copy constructor of 'MoveOnly')|(call to deleted
    // constructor of 'MoveOnly')}}}}
    cuda::std::to_array(cuda::std::move(cmo)); // expected-note {{requested here}}
  }

  return 0;
}
