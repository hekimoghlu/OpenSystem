/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Buffer9.h: Defines the rx::Buffer9 class which implements rx::BufferImpl via rx::BufferD3D.

#ifndef LIBANGLE_RENDERER_D3D_D3D9_BUFFER9_H_
#define LIBANGLE_RENDERER_D3D_D3D9_BUFFER9_H_

#include "common/MemoryBuffer.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/d3d/BufferD3D.h"

namespace rx
{
class Renderer9;

class Buffer9 : public BufferD3D
{
  public:
    Buffer9(const gl::BufferState &state, Renderer9 *renderer);
    ~Buffer9() override;

    // BufferD3D implementation
    size_t getSize() const override;
    bool supportsDirectBinding() const override;
    angle::Result getData(const gl::Context *context, const uint8_t **outData) override;

    // BufferImpl implementation
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
    angle::Result markTransformFeedbackUsage(const gl::Context *context) override;

  private:
    angle::MemoryBuffer mMemory;
    size_t mSize;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D9_BUFFER9_H_
