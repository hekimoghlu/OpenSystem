/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// VertexBuffer11.h: Defines the D3D11 VertexBuffer implementation.

#ifndef LIBANGLE_RENDERER_D3D_D3D11_VERTEXBUFFER11_H_
#define LIBANGLE_RENDERER_D3D_D3D11_VERTEXBUFFER11_H_

#include <stdint.h>

#include "libANGLE/renderer/d3d/VertexBuffer.h"
#include "libANGLE/renderer/d3d/d3d11/ResourceManager11.h"

namespace rx
{
class Renderer11;

class VertexBuffer11 : public VertexBuffer
{
  public:
    explicit VertexBuffer11(Renderer11 *const renderer);

    angle::Result initialize(const gl::Context *context,
                             unsigned int size,
                             bool dynamicUsage) override;

    // Warning: you should ensure binding really matches attrib.bindingIndex before using this
    // function.
    angle::Result storeVertexAttributes(const gl::Context *context,
                                        const gl::VertexAttribute &attrib,
                                        const gl::VertexBinding &binding,
                                        gl::VertexAttribType currentValueType,
                                        GLint start,
                                        size_t count,
                                        GLsizei instances,
                                        unsigned int offset,
                                        const uint8_t *sourceData) override;

    unsigned int getBufferSize() const override;
    angle::Result setBufferSize(const gl::Context *context, unsigned int size) override;
    angle::Result discard(const gl::Context *context) override;

    void hintUnmapResource() override;

    const d3d11::Buffer &getBuffer() const;

  private:
    ~VertexBuffer11() override;
    angle::Result mapResource(const gl::Context *context);

    Renderer11 *const mRenderer;

    d3d11::Buffer mBuffer;
    unsigned int mBufferSize;
    bool mDynamicUsage;

    uint8_t *mMappedResourceData;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D11_VERTEXBUFFER11_H_
