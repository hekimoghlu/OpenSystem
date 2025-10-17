/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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
// BufferNULL.h:
//    Defines the class interface for BufferNULL, implementing BufferImpl.
//

#ifndef LIBANGLE_RENDERER_NULL_BUFFERNULL_H_
#define LIBANGLE_RENDERER_NULL_BUFFERNULL_H_

#include "libANGLE/renderer/BufferImpl.h"

namespace rx
{

class AllocationTrackerNULL;

class BufferNULL : public BufferImpl
{
  public:
    BufferNULL(const gl::BufferState &state, AllocationTrackerNULL *allocationTracker);
    ~BufferNULL() override;

    angle::Result setDataWithUsageFlags(const gl::Context *context,
                                        gl::BufferBinding target,
                                        GLeglClientBufferEXT clientBuffer,
                                        const void *data,
                                        size_t size,
                                        gl::BufferUsage usage,
                                        GLbitfield flags) override;
    angle::Result setData(const gl::Context *context,
                          gl::BufferBinding target,
                          const void *data,
                          size_t size,
                          gl::BufferUsage usage) override;
    angle::Result setSubData(const gl::Context *context,
                             gl::BufferBinding target,
                             const void *data,
                             size_t size,
                             size_t offset) override;
    angle::Result copySubData(const gl::Context *context,
                              BufferImpl *source,
                              GLintptr sourceOffset,
                              GLintptr destOffset,
                              GLsizeiptr size) override;
    angle::Result map(const gl::Context *context, GLenum access, void **mapPtr) override;
    angle::Result mapRange(const gl::Context *context,
                           size_t offset,
                           size_t length,
                           GLbitfield access,
                           void **mapPtr) override;
    angle::Result unmap(const gl::Context *context, GLboolean *result) override;

    angle::Result getIndexRange(const gl::Context *context,
                                gl::DrawElementsType type,
                                size_t offset,
                                size_t count,
                                bool primitiveRestartEnabled,
                                gl::IndexRange *outRange) override;

    uint8_t *getDataPtr();
    const uint8_t *getDataPtr() const;

  private:
    std::vector<uint8_t> mData;

    AllocationTrackerNULL *mAllocationTracker;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_NULL_BUFFERNULL_H_
