/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// BufferNULL.cpp:
//    Implements the class methods for BufferNULL.
//

#include "libANGLE/renderer/null/BufferNULL.h"

#include "common/debug.h"
#include "common/utilities.h"
#include "libANGLE/Context.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/null/ContextNULL.h"

namespace rx
{

BufferNULL::BufferNULL(const gl::BufferState &state, AllocationTrackerNULL *allocationTracker)
    : BufferImpl(state), mAllocationTracker(allocationTracker)
{
    ASSERT(mAllocationTracker != nullptr);
}

BufferNULL::~BufferNULL()
{
    bool memoryReleaseResult = mAllocationTracker->updateMemoryAllocation(mData.size(), 0);
    ASSERT(memoryReleaseResult);
}

angle::Result BufferNULL::setDataWithUsageFlags(const gl::Context *context,
                                                gl::BufferBinding target,
                                                GLeglClientBufferEXT clientBuffer,
                                                const void *data,
                                                size_t size,
                                                gl::BufferUsage usage,
                                                GLbitfield flags)
{
    ANGLE_CHECK_GL_ALLOC(GetImplAs<ContextNULL>(context),
                         mAllocationTracker->updateMemoryAllocation(mData.size(), size));

    mData.resize(size, 0);
    if (size > 0 && data != nullptr)
    {
        memcpy(mData.data(), data, size);
    }
    return angle::Result::Continue;
}

angle::Result BufferNULL::setData(const gl::Context *context,
                                  gl::BufferBinding target,
                                  const void *data,
                                  size_t size,
                                  gl::BufferUsage usage)
{
    ANGLE_CHECK_GL_ALLOC(GetImplAs<ContextNULL>(context),
                         mAllocationTracker->updateMemoryAllocation(mData.size(), size));

    mData.resize(size, 0);
    if (size > 0 && data != nullptr)
    {
        memcpy(mData.data(), data, size);
    }
    return angle::Result::Continue;
}

angle::Result BufferNULL::setSubData(const gl::Context *context,
                                     gl::BufferBinding target,
                                     const void *data,
                                     size_t size,
                                     size_t offset)
{
    if (size > 0)
    {
        memcpy(mData.data() + offset, data, size);
    }
    return angle::Result::Continue;
}

angle::Result BufferNULL::copySubData(const gl::Context *context,
                                      BufferImpl *source,
                                      GLintptr sourceOffset,
                                      GLintptr destOffset,
                                      GLsizeiptr size)
{
    BufferNULL *sourceNULL = GetAs<BufferNULL>(source);
    if (size > 0)
    {
        memcpy(mData.data() + destOffset, sourceNULL->mData.data() + sourceOffset, size);
    }
    return angle::Result::Continue;
}

angle::Result BufferNULL::map(const gl::Context *context, GLenum access, void **mapPtr)
{
    *mapPtr = mData.data();
    return angle::Result::Continue;
}

angle::Result BufferNULL::mapRange(const gl::Context *context,
                                   size_t offset,
                                   size_t length,
                                   GLbitfield access,
                                   void **mapPtr)
{
    *mapPtr = mData.data() + offset;
    return angle::Result::Continue;
}

angle::Result BufferNULL::unmap(const gl::Context *context, GLboolean *result)
{
    *result = GL_TRUE;
    return angle::Result::Continue;
}

angle::Result BufferNULL::getIndexRange(const gl::Context *context,
                                        gl::DrawElementsType type,
                                        size_t offset,
                                        size_t count,
                                        bool primitiveRestartEnabled,
                                        gl::IndexRange *outRange)
{
    *outRange = gl::ComputeIndexRange(type, mData.data() + offset, count, primitiveRestartEnabled);
    return angle::Result::Continue;
}

uint8_t *BufferNULL::getDataPtr()
{
    return mData.data();
}

const uint8_t *BufferNULL::getDataPtr() const
{
    return mData.data();
}

}  // namespace rx
