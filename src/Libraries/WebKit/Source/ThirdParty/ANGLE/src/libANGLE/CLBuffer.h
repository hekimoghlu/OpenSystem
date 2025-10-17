/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
// CLBuffer.h: Defines the cl::Buffer class, which is a collection of elements.

#ifndef LIBANGLE_CLBUFFER_H_
#define LIBANGLE_CLBUFFER_H_

#include "libANGLE/CLMemory.h"

namespace cl
{

class Buffer final : public Memory
{
  public:
    // Front end entry functions, only called from OpenCL entry points

    static bool IsValid(const _cl_mem *buffer);

    cl_mem createSubBuffer(MemFlags flags,
                           cl_buffer_create_type createType,
                           const void *createInfo);

    bool isRegionValid(size_t offset, size_t size) const;
    bool isRegionValid(const cl_buffer_region &region) const;

  public:
    ~Buffer() override;

    MemObjectType getType() const final;

    bool isSubBuffer() const;

  private:
    Buffer(Context &context, PropArray &&properties, MemFlags flags, size_t size, void *hostPtr);

    Buffer(Buffer &parent, MemFlags flags, size_t offset, size_t size);

    friend class Object;
};

inline bool Buffer::IsValid(const _cl_mem *buffer)
{
    return Memory::IsValid(buffer) && buffer->cast<Memory>().getType() == MemObjectType::Buffer;
}

inline bool Buffer::isRegionValid(size_t offset, size_t size) const
{
    return offset < mSize && offset + size <= mSize;
}

inline bool Buffer::isRegionValid(const cl_buffer_region &region) const
{
    return region.origin < mSize && region.origin + region.size <= mSize;
}

inline MemObjectType Buffer::getType() const
{
    return MemObjectType::Buffer;
}

inline bool Buffer::isSubBuffer() const
{
    return mParent != nullptr;
}

}  // namespace cl

#endif  // LIBANGLE_CLBUFFER_H_
