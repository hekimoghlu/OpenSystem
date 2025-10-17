/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// CLObject.h: Defines the cl::Object class, which is the base class of all ANGLE CL objects.

#ifndef LIBANGLE_CLOBJECT_H_
#define LIBANGLE_CLOBJECT_H_

#include "libANGLE/cl_types.h"
#include "libANGLE/renderer/cl_types.h"

#include <atomic>

namespace cl
{

class Object
{
  public:
    Object();
    virtual ~Object();

    cl_uint getRefCount() const noexcept { return mRefCount; }

    void retain() noexcept { ++mRefCount; }

    bool release()
    {
        if (mRefCount == 0u)
        {
            WARN() << "Unreferenced object without references";
            return true;
        }
        return --mRefCount == 0u;
    }

    template <typename T, typename... Args>
    static T *Create(Args &&...args)
    {
        T *object = new T(std::forward<Args>(args)...);
        return object;
    }

  private:
    std::atomic<cl_uint> mRefCount;
};

}  // namespace cl

#endif  // LIBANGLE_CLCONTEXT_H_
