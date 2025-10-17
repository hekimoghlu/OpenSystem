/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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

#ifndef REP_H
#define REP_H

#include "test_macros.h"

class Rep
{
  int data_;

public:
  __host__ __device__ constexpr Rep()
      : data_(-1)
  {}
  __host__ __device__ explicit constexpr Rep(int i)
      : data_(i)
  {}

  __host__ __device__ bool constexpr operator==(int i) const
  {
    return data_ == i;
  }
  __host__ __device__ bool constexpr operator==(const Rep& r) const
  {
    return data_ == r.data_;
  }

  __host__ __device__ Rep& operator*=(Rep x)
  {
    data_ *= x.data_;
    return *this;
  }
  __host__ __device__ Rep& operator/=(Rep x)
  {
    data_ /= x.data_;
    return *this;
  }
};

#endif // REP_H
