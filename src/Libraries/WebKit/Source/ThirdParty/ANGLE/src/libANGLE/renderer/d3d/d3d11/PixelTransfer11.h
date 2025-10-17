/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 30, 2021.
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
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// PixelTransfer11.h:
//   Buffer-to-Texture and Texture-to-Buffer data transfers.
//   Used to implement pixel unpack and pixel pack buffers in ES3.

#ifndef LIBANGLE_RENDERER_D3D_D3D11_PIXELTRANSFER11_H_
#define LIBANGLE_RENDERER_D3D_D3D11_PIXELTRANSFER11_H_

#include <GLES2/gl2.h>

#include <map>

#include <libANGLE/angletypes.h>
#include "common/platform.h"
#include "libANGLE/Error.h"
#include "libANGLE/renderer/d3d/d3d11/ResourceManager11.h"

namespace gl
{
class Buffer;
class Context;
}  // namespace gl

namespace rx
{
class Renderer11;
class RenderTargetD3D;

class PixelTransfer11
{
  public:
    explicit PixelTransfer11(Renderer11 *renderer);
    ~PixelTransfer11();

    // unpack: the source buffer is stored in the unpack state, and buffer strides
    // offset: the start of the data within the unpack buffer
    // destRenderTarget: individual slice/layer of a target texture
    // destinationFormat/sourcePixelsType: determines shaders + shader parameters
    // destArea: the sub-section of destRenderTarget to copy to
    angle::Result copyBufferToTexture(const gl::Context *context,
                                      const gl::PixelUnpackState &unpack,
                                      gl::Buffer *unpackBuffer,
                                      unsigned int offset,
                                      RenderTargetD3D *destRenderTarget,
                                      GLenum destinationFormat,
                                      GLenum sourcePixelsType,
                                      const gl::Box &destArea);

  private:
    struct CopyShaderParams
    {
        unsigned int FirstPixelOffset;
        unsigned int PixelsPerRow;
        unsigned int RowStride;
        unsigned int RowsPerSlice;
        float PositionOffset[2];
        float PositionScale[2];
        int TexLocationOffset[2];
        int TexLocationScale[2];
        unsigned int FirstSlice;
    };

    static void setBufferToTextureCopyParams(const gl::Box &destArea,
                                             const gl::Extents &destSize,
                                             GLenum internalFormat,
                                             const gl::PixelUnpackState &unpack,
                                             unsigned int offset,
                                             CopyShaderParams *parametersOut);

    angle::Result loadResources(const gl::Context *context);
    angle::Result buildShaderMap(const gl::Context *context);
    const d3d11::PixelShader *findBufferToTexturePS(GLenum internalFormat) const;

    Renderer11 *mRenderer;

    bool mResourcesLoaded;
    std::map<GLenum, d3d11::PixelShader> mBufferToTexturePSMap;
    d3d11::VertexShader mBufferToTextureVS;
    d3d11::GeometryShader mBufferToTextureGS;
    d3d11::Buffer mParamsConstantBuffer;
    CopyShaderParams mParamsData;

    d3d11::RasterizerState mCopyRasterizerState;
    d3d11::DepthStencilState mCopyDepthStencilState;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D11_PIXELTRANSFER11_H_
