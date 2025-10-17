/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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

#ifndef __PRIVATE_CONSTRUCTOR__H
#define __PRIVATE_CONSTRUCTOR__H

#include <iostream>

struct PrivateConstructor
{
  PrivateConstructor static make(int v)
  {
    return PrivateConstructor(v);
  }
  int get() const
  {
    return val;
  }

private:
  PrivateConstructor(int v)
      : val(v)
  {}
  int val;
};

bool operator<(const PrivateConstructor& lhs, const PrivateConstructor& rhs)
{
  return lhs.get() < rhs.get();
}

bool operator<(const PrivateConstructor& lhs, int rhs)
{
  return lhs.get() < rhs;
}
bool operator<(int lhs, const PrivateConstructor& rhs)
{
  return lhs < rhs.get();
}

std::ostream& operator<<(std::ostream& os, const PrivateConstructor& foo)
{
  return os << foo.get();
}

#endif
