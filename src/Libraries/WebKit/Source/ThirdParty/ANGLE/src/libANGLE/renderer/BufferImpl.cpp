/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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

// BufferImpl.cpp: Implementation methods rx::BufferImpl class.

#include "libANGLE/renderer/BufferImpl.h"

namespace rx
{

angle::Result BufferImpl::getSubData(const gl::Context *context,
                                     GLintptr offset,
                                     GLsizeiptr size,
                                     void *outData)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

angle::Result BufferImpl::setDataWithUsageFlags(const gl::Context *context,
                                                gl::BufferBinding target,
                                                GLeglClientBufferEXT clientBuffer,
                                                const void *data,
                                                size_t size,
                                                gl::BufferUsage usage,
                                                GLbitfield flags)
{
    return setData(context, target, data, size, usage);
}

angle::Result BufferImpl::onLabelUpdate(const gl::Context *context)
{
    return angle::Result::Continue;
}

}  // namespace rx
