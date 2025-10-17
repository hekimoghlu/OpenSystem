/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 8, 2022.
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

//
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_MSL_REFERENCE_H_
#define COMPILER_TRANSLATOR_MSL_REFERENCE_H_

namespace sh
{

// Similar to std::reference_wrapper, but also lifts comparison operators.
template <typename T>
class Ref
{
  public:
    Ref(const Ref &) = default;
    Ref(Ref &&)      = default;
    Ref(T &ref) : mPtr(&ref) {}

    Ref &operator=(const Ref &) = default;
    Ref &operator=(Ref &&)      = default;

    bool operator==(const Ref &other) const { return *mPtr == *other.mPtr; }
    bool operator!=(const Ref &other) const { return *mPtr != *other.mPtr; }
    bool operator<=(const Ref &other) const { return *mPtr <= *other.mPtr; }
    bool operator>=(const Ref &other) const { return *mPtr >= *other.mPtr; }
    bool operator<(const Ref &other) const { return *mPtr < *other.mPtr; }
    bool operator>(const Ref &other) const { return *mPtr > *other.mPtr; }

    T &get() { return *mPtr; }
    T const &get() const { return *mPtr; }

    operator T &() { return *mPtr; }
    operator T const &() const { return *mPtr; }

  private:
    T *mPtr;
};

template <typename T>
using CRef = Ref<T const>;

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_MSL_REFERENCE_H_
