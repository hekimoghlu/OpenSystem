/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Optional.h:
//   Represents a type that may be invalid, similar to std::optional.
//

#ifndef COMMON_OPTIONAL_H_
#define COMMON_OPTIONAL_H_

#include <utility>

template <class T>
struct Optional
{
    Optional() : mValid(false), mValue(T()) {}

    Optional(const T &valueIn) : mValid(true), mValue(valueIn) {}

    Optional(const Optional &other) : mValid(other.mValid), mValue(other.mValue) {}

    Optional &operator=(const Optional &other)
    {
        this->mValid = other.mValid;
        this->mValue = other.mValue;
        return *this;
    }

    Optional &operator=(const T &value)
    {
        mValue = value;
        mValid = true;
        return *this;
    }

    Optional &operator=(T &&value)
    {
        mValue = std::move(value);
        mValid = true;
        return *this;
    }

    void reset() { mValid = false; }
    T &&release()
    {
        mValid = false;
        return std::move(mValue);
    }

    static Optional Invalid() { return Optional(); }

    bool valid() const { return mValid; }
    T &value() { return mValue; }
    const T &value() const { return mValue; }
    const T &valueOr(const T &defaultValue) const { return mValid ? mValue : defaultValue; }

    bool operator==(const Optional &other) const
    {
        return ((mValid == other.mValid) && (!mValid || (mValue == other.mValue)));
    }

    bool operator!=(const Optional &other) const { return !(*this == other); }

    bool operator==(const T &value) const { return mValid && (mValue == value); }

    bool operator!=(const T &value) const { return !(*this == value); }

  private:
    bool mValid;
    T mValue;
};

#endif  // COMMON_OPTIONAL_H_
