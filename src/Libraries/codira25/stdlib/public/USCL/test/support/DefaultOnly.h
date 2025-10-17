/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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

#ifndef DEFAULTONLY_H
#define DEFAULTONLY_H

#include <uscl/std/cassert>

#include "test_macros.h"

class DefaultOnly
{
  int data_;

  __host__ __device__ DefaultOnly(const DefaultOnly&);
  __host__ __device__ DefaultOnly& operator=(const DefaultOnly&);

public:
  STATIC_MEMBER_VAR(count, int)

  __host__ __device__ DefaultOnly()
      : data_(-1)
  {
    ++count();
  }
  __host__ __device__ ~DefaultOnly()
  {
    data_ = 0;
    --count();
  }

  __host__ __device__ friend bool operator==(const DefaultOnly& x, const DefaultOnly& y)
  {
    return x.data_ == y.data_;
  }
  __host__ __device__ friend bool operator<(const DefaultOnly& x, const DefaultOnly& y)
  {
    return x.data_ < y.data_;
  }
};

#endif // DEFAULTONLY_H
