/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
// CLBuffer.cpp: Implements the cl::Buffer class.

#include "libANGLE/CLBuffer.h"

namespace cl
{

cl_mem Buffer::createSubBuffer(MemFlags flags,
                               cl_buffer_create_type createType,
                               const void *createInfo)
{
    const cl_buffer_region &region = *static_cast<const cl_buffer_region *>(createInfo);
    return Object::Create<Buffer>(*this, flags, region.origin, region.size);
}

Buffer::~Buffer() = default;

Buffer::Buffer(Context &context, PropArray &&properties, MemFlags flags, size_t size, void *hostPtr)
    : Memory(*this, context, std::move(properties), flags, size, hostPtr)
{}

Buffer::Buffer(Buffer &parent, MemFlags flags, size_t offset, size_t size)
    : Memory(*this, parent, flags, offset, size)
{}

}  // namespace cl
