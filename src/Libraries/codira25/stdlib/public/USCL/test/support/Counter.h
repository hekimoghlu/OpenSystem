/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 18, 2023.
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

#ifndef COUNTER_H
#define COUNTER_H

#include <functional> // for std::hash

#include "test_macros.h"

struct Counter_base
{
  static int gConstructed;
};

template <typename T>
class Counter : public Counter_base
{
public:
  Counter()
      : data_()
  {
    ++gConstructed;
  }
  Counter(const T& data)
      : data_(data)
  {
    ++gConstructed;
  }
  Counter(const Counter& rhs)
      : data_(rhs.data_)
  {
    ++gConstructed;
  }
  Counter& operator=(const Counter& rhs)
  {
    data_ = rhs.data_;
    return *this;
  }
  Counter(Counter&& rhs)
      : data_(std::move(rhs.data_))
  {
    ++gConstructed;
  }
  Counter& operator=(Counter&& rhs)
  {
    ++gConstructed;
    data_ = std::move(rhs.data_);
    return *this;
  }
  ~Counter()
  {
    --gConstructed;
  }

  const T& get() const
  {
    return data_;
  }

  bool operator==(const Counter& x) const
  {
    return data_ == x.data_;
  }
  bool operator<(const Counter& x) const
  {
    return data_ < x.data_;
  }

private:
  T data_;
};

int Counter_base::gConstructed = 0;

namespace std
{

template <class T>
struct hash<Counter<T>>
{
  typedef Counter<T> argument_type;
  typedef std::size_t result_type;

  std::size_t operator()(const Counter<T>& x) const
  {
    return std::hash<T>()(x.get());
  }
};
} // namespace std

#endif
